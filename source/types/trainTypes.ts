// ------------------------------- Training Configuration ----------------------------------
export type OptimizerConfig = {
    lr: number;
    eps?: number;
    weight_decay?: number;
    betas?: number[];
    rho?: number;
    beta2_decay?: number;
    d?: number;
    lambd?: number;
    alpha?: number;
    t0?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    etas?: number[]
    step_sizes?: number[];
    max_iter?: number;
    max_eval?: number;
    tolerance_grad?: number;
    tolerance_change?: number;
    history_size?: number;
};  

export type LossFunctionConfig =  {
    reduction?: string;
    ignore_index?: number;
    label_smoothing?: number;
};

export type TrainMetrics  = {
    gradient_visualization: boolean;
    gradient_visualization_period?: number;
    gradient_norm_visualization: boolean;
    gradient_norm_visualization_period?: number;
    learning_rate_visualization: boolean;
    learning_rate_visualization_period?: number;
    weights_visualization: boolean;
    weights_visualization_period?: number;
    graph_visualization: boolean;
    profile: boolean;
    visualize_accuracy: boolean;
    visualize_loss: boolean;
    test_validation: boolean;
    test_validation_period: number;
    train_validation: boolean;
    train_validation_period: number;
}

export type TrainConfig = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfig;
    loss_function_config?: LossFunctionConfig;
    loss_function: string;
    metrics: TrainMetrics;
};

export type OptimizerConfigInput = {
    lr: number;
    eps?: number;
    weight_decay?: number;
    betas?: number[];
    rho?: number;
    beta2_decay?: number;
    d?: number;
    lambd?: number;
    alpha?: number;
    t0?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    etas?: number[]
    step_sizes?: number[];
    max_iter?: number;
    max_eval?: number;
    tolerance_grad?: number;
    tolerance_change?: number;
    history_size?: number;
};

export type TrainMetricsInput  = {
    gradient_visualization: boolean;
    gradient_visualization_period?: number;
    gradient_norm_visualization: boolean;
    gradient_norm_visualization_period?: number;
    learning_rate_visualization: boolean;
    learning_rate_visualization_period?: number;
    weights_visualization: boolean;
    weights_visualization_period?: number;
    graph_visualization: boolean;
    profile: boolean;
    visualize_accuracy: boolean;
    visualize_loss: boolean;
    test_validation: boolean;
    test_validation_period: number;
    train_validation: boolean;
    train_validation_period: number;
}

export type LossFunctionConfigInput =  {
    reduction?: string;
    ignore_index?: number;
    label_smoothing?: number;
};

export type TrainConfigInput = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfigInput;
    loss_function_config?: LossFunctionConfigInput;
    loss_function: string;
    metrics: TrainMetricsInput
};

export type TrainStatus = {
    epoch: number;
    loss: number;
    accuracy: number;
    started: boolean;
    completed: boolean;
    timestamp?: string;
}

