import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.average = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        front = x[:, :1].repeat(1, padding, 1)
        end = x[:, -1:].repeat(1, padding, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.average(x.transpose(1, 2))
        return x.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class SeasonalLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layer_norm(x)
        bias = x.mean(dim=1, keepdim=True)
        return x - bias


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.projection = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.projection.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.projection(x.transpose(1, 2)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, input_dim, external_dim, d_model, dropout):
        super().__init__()
        self.value_embedding = TokenEmbedding(input_dim, d_model)
        self.external_embedding = nn.Linear(external_dim, d_model, bias=False) if external_dim > 0 else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, external=None):
        output = self.value_embedding(values)
        if self.external_embedding is not None:
            output = output + self.external_embedding(external)
        return self.dropout(output)


class AutoCorrelation(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def _top_k(self, length):
        return max(1, min(length, int(self.factor * math.log(length))))

    def _training_aggregation(self, values, correlation):
        B, H, D, L = values.shape
        top_k = self._top_k(L)
        delay_scores = correlation.mean(dim=(1, 2))
        delays = delay_scores.mean(dim=0).topk(top_k).indices
        selected_scores = torch.stack([delay_scores[:, delay] for delay in delays], dim=-1)
        weights = F.softmax(selected_scores, dim=-1)
        output = torch.zeros_like(values)

        for index in range(top_k):
            shifted = torch.roll(values, -int(delays[index]), dims=-1)
            output = output + shifted * weights[:, index].view(B, 1, 1, 1)

        return output

    def _inference_aggregation(self, values, correlation):
        B, H, D, L = values.shape
        top_k = self._top_k(L)
        delay_scores = correlation.mean(dim=(1, 2))
        selected_scores, delays = delay_scores.topk(top_k, dim=-1)
        weights = F.softmax(selected_scores, dim=-1)
        positions = torch.arange(L, device=values.device).view(1, 1, L)
        indices = positions + delays.unsqueeze(-1)
        indices = indices[:, None, None].expand(B, H, D, top_k, L)
        repeated_values = values.repeat(1, 1, 1, 2)
        repeated_values = repeated_values.unsqueeze(3).expand(B, H, D, top_k, 2 * L)
        shifted = torch.gather(repeated_values, dim=-1, index=indices)
        return (shifted * weights[:, None, None, :, None]).sum(dim=3)

    def forward(self, queries, keys, values):
        B, L, H, D = queries.shape
        source_length = values.shape[1]

        if L > source_length:
            padding = queries.new_zeros(B, L - source_length, H, D)
            keys = torch.cat([keys, padding], dim=1)
            values = torch.cat([values, padding], dim=1)
        else:
            keys = keys[:, :L]
            values = values[:, :L]

        queries = queries.permute(0, 2, 3, 1).contiguous()
        keys = keys.permute(0, 2, 3, 1).contiguous()
        values = values.permute(0, 2, 3, 1).contiguous()
        query_spectrum = torch.fft.rfft(queries, dim=-1)
        key_spectrum = torch.fft.rfft(keys, dim=-1)
        correlation = torch.fft.irfft(query_spectrum * key_spectrum.conj(), n=L, dim=-1)

        if self.training:
            output = self._training_aggregation(values, correlation)
        else:
            output = self._inference_aggregation(values, correlation)

        return output.permute(0, 3, 1, 2).contiguous()


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor):
        super().__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.correlation = AutoCorrelation(factor)

    def forward(self, queries, keys, values):
        B, L, d_model = queries.shape
        source_length = keys.shape[1]
        head_dim = d_model // self.n_heads
        queries = self.query_projection(queries).view(B, L, self.n_heads, head_dim)
        keys = self.key_projection(keys).view(B, source_length, self.n_heads, head_dim)
        values = self.value_projection(values).view(B, source_length, self.n_heads, head_dim)
        output = self.correlation(queries, keys, values)
        output = output.view(B, L, d_model)
        return self.output_projection(output)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.auto_correlation = AutoCorrelationLayer(config.d_model, config.n_heads, config.delay_factor)
        self.decomposition1 = SeriesDecomposition(config.moving_avg)
        self.decomposition2 = SeriesDecomposition(config.moving_avg)
        self.conv1 = nn.Conv1d(config.d_model, config.d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(config.d_ff, config.d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = F.gelu if config.activation == "gelu" else F.relu

    def forward(self, x):
        x = x + self.dropout(self.auto_correlation(x, x, x))
        x, _ = self.decomposition1(x)
        output = self.conv1(x.transpose(1, 2))
        output = self.dropout(self.activation(output))
        output = self.dropout(self.conv2(output)).transpose(1, 2)
        x, _ = self.decomposition2(x + output)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.norm = SeasonalLayerNorm(config.d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_correlation = AutoCorrelationLayer(config.d_model, config.n_heads, config.delay_factor)
        self.cross_correlation = AutoCorrelationLayer(config.d_model, config.n_heads, config.delay_factor)
        self.decomposition1 = SeriesDecomposition(config.moving_avg)
        self.decomposition2 = SeriesDecomposition(config.moving_avg)
        self.decomposition3 = SeriesDecomposition(config.moving_avg)
        self.conv1 = nn.Conv1d(config.d_model, config.d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(config.d_ff, config.d_model, kernel_size=1, bias=False)
        self.trend_projection = nn.Conv1d(config.d_model, config.output_dim, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = F.gelu if config.activation == "gelu" else F.relu

    def forward(self, x, encoder_output):
        x = x + self.dropout(self.self_correlation(x, x, x))
        x, trend1 = self.decomposition1(x)
        x = x + self.dropout(self.cross_correlation(x, encoder_output, encoder_output))
        x, trend2 = self.decomposition2(x)
        output = self.conv1(x.transpose(1, 2))
        output = self.dropout(self.activation(output))
        output = self.dropout(self.conv2(output)).transpose(1, 2)
        x, trend3 = self.decomposition3(x + output)
        trend = trend1 + trend2 + trend3
        trend = self.trend_projection(trend.transpose(1, 2)).transpose(1, 2)
        return x, trend


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(*[DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.norm = SeasonalLayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, config.output_dim)

    def forward(self, x, encoder_output, trend):
        for layer in self.layers:
            x, residual_trend = layer(x, encoder_output)
            trend = trend + residual_trend
        seasonal = self.projection(self.norm(x))
        return seasonal, trend


class Autoformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decomposition = SeriesDecomposition(config.moving_avg)
        self.encoder_embedding = DataEmbedding(1, config.external_dim, config.d_model, config.dropout)
        self.decoder_embedding = DataEmbedding(1, config.external_dim, config.d_model, config.dropout)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def _prepare_values(self, values):
        if values.ndim == 2:
            values = values.unsqueeze(-1)
        return values

    def _prepare_external(self, past_external, future_external):
        if self.config.external_dim == 0:
            return None
        return torch.cat([past_external[:, -self.config.label_length:], future_external], dim=1)

    def forward(self, past_values, past_external=None, future_external=None):
        past_values = self._prepare_values(past_values)
        B = past_values.shape[0]
        decoder_external = self._prepare_external(past_external, future_external)
        seasonal, trend = self.decomposition(past_values)
        context_mean = past_values.mean(dim=1, keepdim=True).repeat(1, self.config.pred_length, 1)
        zeros = past_values.new_zeros(B, self.config.pred_length, 1)
        seasonal_init = torch.cat([seasonal[:, -self.config.label_length:], zeros], dim=1)
        trend_init = torch.cat([trend[:, -self.config.label_length:], context_mean], dim=1)
        encoder_input = self.encoder_embedding(past_values, past_external)
        encoder_output = self.encoder(encoder_input)
        decoder_input = self.decoder_embedding(seasonal_init, decoder_external)
        seasonal_output, trend_output = self.decoder(decoder_input, encoder_output, trend_init)
        forecast = seasonal_output + trend_output
        return forecast[:, -self.config.pred_length:]
