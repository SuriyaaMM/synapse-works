import type { LayerConfigInput } from "../../../../../../source/types/layerTypes";

export const LayerConfigMap = new Map<string, LayerConfigInput>([
  ['linear', { type: 'linear', linear: { in_features: 0, out_features: 0 } }],
  ['conv2d', { type: 'conv2d', conv2d: { in_channels: 0, out_channels: 0, kernel_size: [] } }],
  ['convtranspose2d', { type: 'convtranspose2d', convtranspose2d: { in_channels: 0, out_channels: 0, kernel_size: [] } }],
  ['conv1d', { type: 'conv1d', conv1d: { in_channels: 0, out_channels: 0, kernel_size: [] } }],
  ['maxpool2d', { type: 'maxpool2d', maxpool2d: { kernel_size: [] } }],
  ['maxpool1d', { type: 'maxpool1d', maxpool1d: { kernel_size: [] } }],
  ['avgpool2d', { type: 'avgpool2d', avgpool2d: { kernel_size: [] } }],
  ['avgpool1d', { type: 'avgpool1d', avgpool1d: { kernel_size: [] } }],
  ['batchnorm2d', { type: 'batchnorm2d', batchnorm2d: { num_features: 0 } }],
  ['batchnorm1d', { type: 'batchnorm1d', batchnorm1d: { num_features: 0 } }],
  ['flatten', { type: 'flatten', flatten: {} }],
  ['dropout', { type: 'dropout', dropout: {} }],
  ['dropout2d', { type: 'dropout2d', dropout2d: {} }],
  ['elu', { type: 'elu', elu: {} }],
  ['relu', { type: 'relu', relu: {} }],
  ['leakyrelu', { type: 'leakyrelu', leakyrelu: {} }],
  ['sigmoid', { type: 'sigmoid', sigmoid: {} }],
  ['logsigmoid', { type: 'logsigmoid', logsigmoid: {} }],
  ['tanh', { type: 'tanh', tanh: {} }],
  ['cat', { type: 'cat', cat: {} }]
]);

export function convertLayerToConfig(layer: any): LayerConfigInput {
  console.log("5. layertype:", layer);
  const template = LayerConfigMap.get(layer.type);

  if (!template) {
    throw new Error(`Unsupported layer type: ${layer.type}`);
  }

  const config: LayerConfigInput = JSON.parse(JSON.stringify(template));

  switch (layer.type) {
    case 'linear':
      config.linear = {
        name: layer.name,
        in_features: layer.in_features,
        out_features: layer.out_features,
        bias: layer.bias ?? true
      };
      break;

    case 'conv2d':
      config.conv2d = {
        name: layer.name,
        in_channels: layer.in_channels,
        out_channels: layer.out_channels,
        kernel_size: layer.kernel_size,
        stride: layer.stride,
        padding: layer.padding,
        dilation: layer.dilation,
        groups: layer.groups,
        bias: layer.bias ?? true,
        padding_mode: layer.padding_mode || 'zeros'
      };
      break;

    case 'convtranspose2d':
      config.convtranspose2d = {
        name: layer.name,
        in_channels: layer.in_channels,
        out_channels: layer.out_channels,
        kernel_size: layer.kernel_size,
        stride: layer.stride,
        padding: layer.padding,
        dilation: layer.dilation,
        groups: layer.groups,
        bias: layer.bias ?? true,
        output_padding: layer.output_padding
      };
      break;

    case 'conv1d':
      config.conv1d = {
        name: layer.name,
        in_channels: layer.in_channels,
        out_channels: layer.out_channels,
        kernel_size: layer.kernel_size,
        stride: layer.stride,
        padding: layer.padding,
        dilation: layer.dilation,
        groups: layer.groups,
        bias: layer.bias ?? true,
        padding_mode: layer.padding_mode || 'zeros'
      };
      break;

    case 'maxpool2d':
      config.maxpool2d = {
        name: layer.name,
        kernel_size: layer.kernel_size,
        stride: layer.stride,
        padding: layer.padding,
        dilation: layer.dilation,
        return_indices: layer.return_indices ?? false,
        ceil_mode: layer.ceil_mode ?? false
      };
      break;

    case 'maxpool1d':
      config.maxpool1d = {
        name: layer.name,
        kernel_size: layer.kernel_size,
        stride: layer.stride,
        padding: layer.padding,
        dilation: layer.dilation,
        return_indices: layer.return_indices ?? false,
        ceil_mode: layer.ceil_mode ?? false
      };
      break;

    case 'avgpool2d':
      config.avgpool2d = {
        name: layer.name,
        kernel_size: layer.kernel_size,
        stride: layer.stride,
        padding: layer.padding,
        count_include_pad: layer.count_include_pad ?? false,
        divisor_override: layer.divisor_override,
        ceil_mode: layer.ceil_mode ?? false
      };
      break;

    case 'avgpool1d':
      config.avgpool1d = {
        name: layer.name,
        kernel_size: layer.kernel_size,
        stride: layer.stride,
        padding: layer.padding,
        count_include_pad: layer.count_include_pad ?? false,
        divisor_override: layer.divisor_override,
        ceil_mode: layer.ceil_mode ?? false
      };
      break;

    case 'batchnorm2d':
      config.batchnorm2d = {
        name: layer.name,
        num_features: layer.num_features,
        eps: layer.eps || 1e-5,
        momentum: layer.momentum || 0.1,
        affine: layer.affine ?? true,
        track_running_status: layer.track_running_stats ?? true
      };
      break;

    case 'batchnorm1d':
      config.batchnorm1d = {
        name: layer.name,
        num_features: layer.num_features,
        eps: layer.eps || 1e-5,
        momentum: layer.momentum || 0.1,
        affine: layer.affine ?? true,
        track_running_status: layer.track_running_stats ?? true
      };
      break;

    case 'flatten':
      config.flatten = {
        name: layer.name,
        start_dim: layer.start_dim || 1,
        end_dim: layer.end_dim || -1
      };
      break;

    case 'dropout':
      config.dropout = {
        name: layer.name,
        p: layer.p || 0.5
      };
      break;

    case 'dropout2d':
      config.dropout2d = {
        name: layer.name,
        p: layer.p || 0.5
      };
      break;

    case 'elu':
      config.elu = {
        name: layer.name,
        alpha: layer.alpha || 1.0,
        inplace: layer.inplace ?? false
      };
      break;

    case 'relu':
      config.relu = {
        name: layer.name,
        inplace: layer.inplace ?? false
      };
      break;

    case 'leakyrelu':
      config.leakyrelu = {
        name: layer.name,
        negative_slope: layer.negative_slope || 0.01,
        inplace: layer.inplace ?? false
      };
      break;

    case 'sigmoid':
      config.sigmoid = {
        name: layer.name
      };
      break;

    case 'logsigmoid':
      config.logsigmoid = {
        name: layer.name
      };
      break;

    case 'tanh':
      config.tanh = {
        name: layer.name
      };
      break;

    case 'cat':
      config.cat = {
        name: layer.name,
        dimension: layer.dimension || 0
      };
      break;

    default:
      throw new Error(`Unsupported layer type: ${layer.type}`);
  }

  return config;
}

