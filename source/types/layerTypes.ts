// ------------------------------- Layers & Configuration ----------------------------------
export interface LayerConfig {
    id: string;
    type: string;
    name?: string;
    in_dimension?: number[];
    out_dimension?: number[];
};

export interface LinearLayerConfig extends LayerConfig {
    in_features: number;
    out_features: number;
    bias?: boolean;
};

export interface Conv2dLayerConfig extends LayerConfig {
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
};

export interface ConvTranspose2dLayerConfig extends LayerConfig {
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    output_padding?: number[];
};

export interface Conv1dLayerConfig extends LayerConfig {
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
};

export interface MaxPool2dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export interface MaxPool1dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export interface AvgPool2dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export interface AvgPool1dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export interface BatchNorm2dLayerConfig extends LayerConfig {
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status?:boolean;
}

export interface BatchNorm1dLayerConfig extends LayerConfig {
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status: boolean;
}

export interface FlattenLayerConfig extends LayerConfig {
    start_dim?: number;
    end_dim?: number;
}

export interface DropoutLayerConfig extends LayerConfig {
    p?: number;
}

export interface Dropout2dLayerConfig extends LayerConfig {
    p?: number;
}

export interface ELULayerConfig extends LayerConfig {
    alpha?: number;
    inplace?: boolean;
}

export interface ReLULayerConfig extends LayerConfig {
    inplace?: boolean;
}

export interface LeakyReLULayerConfig extends LayerConfig {
    negative_slope?: number;
    inplace?: boolean;
}


export interface SigmoidLayerConfig extends LayerConfig {}
export interface LogSigmoidLayerConfig extends LayerConfig {}
export interface TanhLayerConfig extends LayerConfig {}

export interface CatLayerConfig extends LayerConfig {
    dimension?: number;
}

export type LinearLayerConfigInput = {
    name?: string;
    in_features: number;
    out_features: number;
    bias?: boolean;
};

export type Conv2dLayerConfigInput = {
    name?: string;
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
}

export type ConvTranspose2dLayerConfigInput = {
    name?: string;
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    output_padding?: number[];
};

export type Conv1dLayerConfigInput  = {
    name?: string;
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
};

export type MaxPool2dLayerConfigInput = {
    name?: string;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export type MaxPool1dLayerConfigInput = {
    name?: string;
    kernel_size: number[];
    stride?: number[]
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export type AvgPool2dLayerConfigInput =  {
    name? : string;
    kernel_size: number[];
    stride?: number[]
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export type AvgPool1dLayerConfigInput =  {
    name? : string;
    kernel_size: number[];
    stride?: number[]
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export type BatchNorm2dLayerConfigInput = {
    name?: string;
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status?: boolean;
}

export type BatchNorm1dLayerConfigInput = {
    name?: string;
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status?: boolean;
}

export type FlattenLayerConfigInput = {
    name?: string;
    start_dim?: number;
    end_dim?: number;
}

export type DropoutLayerConfigInput = {
    name?: string;
    p?: number;
}

export type Dropout2dLayerConfigInput = {
    name?: string;
    p?: number;
}

export type ELULayerConfigInput = {
    name?: string;
    alpha?: number;
    inplace?: boolean;
}

export type ReLULayerConfigInput = {
    name?: string;
    inplace?: boolean;
}

export type LeakyReLULayerConfigInput = {
    name?: string;
    negative_slope?: number;
    inplace?: boolean;
}

export type SigmoidLayerConfigInput = {
    name?: string;
}

export type LogSigmoidLayerConfigInput = {
    name?: string;
}

export type TanhLayerConfigInput = {
    name?: string;
}

export type CatLayerConfigInput = {
    name?: string;
    dimension?: number;
}

export type LayerConfigInput = {
    type: string; 
    linear?: LinearLayerConfigInput;
    conv2d?: Conv2dLayerConfigInput;
    convtranspose2d?: ConvTranspose2dLayerConfigInput;
    conv1d?: Conv1dLayerConfigInput;
    maxpool2d?: MaxPool2dLayerConfigInput;
    maxpool1d?: MaxPool1dLayerConfigInput;
    avgpool2d?: AvgPool2dLayerConfigInput;
    avgpool1d?: AvgPool1dLayerConfigInput;
    batchnorm2d?: BatchNorm2dLayerConfigInput;
    batchnorm1d?: BatchNorm1dLayerConfigInput;
    flatten?: FlattenLayerConfigInput;
    dropout?: DropoutLayerConfigInput;
    dropout2d?: Dropout2dLayerConfigInput;
    elu?: ELULayerConfigInput;
    relu?: ReLULayerConfigInput;
    leakyrelu?: LeakyReLULayerConfigInput;
    sigmoid?: SigmoidLayerConfigInput;
    logsigmoid?: LogSigmoidLayerConfigInput;
    tanh?: TanhLayerConfigInput;
    cat?: CatLayerConfigInput;
};
