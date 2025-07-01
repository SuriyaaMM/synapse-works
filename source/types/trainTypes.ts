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

export type TrainConfig = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfig;
    loss_function: string;
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

export type TrainConfigInput = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfigInput;
    loss_function: string;
};

export type TrainStatus = {
    epoch: number;
    loss: number;
    accuracy: number;
    started: boolean;
    completed: boolean;
    timestamp?: string;
}

