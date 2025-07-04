export interface ParamConfig {
  type: 'number' | 'boolean' | 'array';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  label: string;
  required: boolean;
  format?: 'scientific';
  subtype?: 'number';
}

export interface OptimizerSchema {
  [key: string]: ParamConfig;
}

export const optimizerConfigs: Record<string, OptimizerSchema> = {
  adadelta: {
    lr: { type: 'number', default: 1.0, min: 0.001, max: 10, step: 0.001, label: 'Learning Rate', required: true },
    rho: { type: 'number', default: 0.9, min: 0, max: 1, step: 0.01, label: 'Rho (ρ) - decay rate', required: false },
    eps: { type: 'number', default: 1e-6, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
    weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
  },
  adafactor: {
    lr: { type: 'number', default: 1e-2, min: 1e-5, max: 1, step: 1e-5, label: 'Learning Rate', format: 'scientific', required: true },
    eps: { type: 'number', default: 1e-3, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
    weight_decay: { type: 'number', default: 1e-2, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
    beta2_decay: { type: 'number', default: -0.8, min: -1, max: 0, step: 0.01, label: 'Beta2 Decay (for squared gradient averaging)', required: false },
    d: { type: 'number', default: 1.0, min: 0.1, max: 10, step: 0.1, label: 'Clipping Threshold (d)', required: false }
  },
  adam: {
    lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
    weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
    betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false }
  },
  adamw: {
    lr: { type: 'number', default: 1e-3, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
    weight_decay: { type: 'number', default: 0.01, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
    betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false }
  },
  sparseadam: {
    lr: { type: 'number', default: 0.001, min: 1e-5, max: 1, step: 1e-5, label: 'Learning Rate', format: 'scientific', required: true },
    betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false },
    eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false }
  },
  adamax: {
    lr: { type: 'number', default: 0.002, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false },
    eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
    weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
  },
  asgd: {
    lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    lambd: { type: 'number', default: 0.0001, min: 0, max: 1, step: 0.0001, label: 'Lambda (λ)', required: false },
    alpha: { type: 'number', default: 0.75, min: 0, max: 1, step: 0.01, label: 'Alpha (α)', required: false },
    t0: { type: 'number', default: 1000000, min: 1, max: 10000000, step: 1, label: 'T0 (averaging start point)', required: false },
    weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
  },
  lbfgs: {
    lr: { type: 'number', default: 1.0, min: 0.001, max: 10, step: 0.001, label: 'Learning Rate', required: true },
    max_iter: { type: 'number', default: 20, min: 1, max: 1000, step: 1, label: 'Max Iterations', required: false },
    max_eval: { type: 'number', default: 25, min: 1, max: 1000, step: 1, label: 'Max Function Evaluations', required: false },
    tolerance_grad: { type: 'number', default: 1e-7, min: 1e-12, max: 1e-3, step: 1e-8, label: 'Gradient Tolerance', format: 'scientific', required: false },
    tolerance_change: { type: 'number', default: 1e-9, min: 1e-15, max: 1e-3, step: 1e-10, label: 'Change Tolerance', format: 'scientific', required: false },
    history_size: { type: 'number', default: 100, min: 1, max: 1000, step: 1, label: 'History Size', required: false }
  },
  radam: {
    lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false },
    eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
    weight_decay: { type: 'number', default: 0.0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
  },
  rmsprop: {
    lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    alpha: { type: 'number', default: 0.99, min: 0, max: 1, step: 0.01, label: 'Alpha (smoothing constant)', required: false },
    eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
    weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
    momentum: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Momentum', required: false }
  },
  rprop: {
    lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    etas: { type: 'array', default: [0.5, 1.2], label: 'Eta Parameters (η-, η+)', subtype: 'number', min: 0.1, max: 2, step: 0.1, required: false },
    step_sizes: { type: 'array', default: [1e-6, 50], label: 'Step Size Range (min, max)', subtype: 'number', min: 1e-8, max: 100, step: 0.00000001, required: false }
  },
  sgd: {
    lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
    momentum: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Momentum', required: false },
    dampening: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Dampening', required: false },
    weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
    nesterov: { type: 'boolean', default: false, label: 'Nesterov Momentum', required: false }
  }
};

export const optimizerOptions = [
  { value: '', label: 'Select an optimizer...' },
  { value: 'adadelta', label: 'Adadelta' },
  { value: 'adafactor', label: 'Adafactor' },
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'sparseadam', label: 'SparseAdam' },
  { value: 'adamax', label: 'Adamax' },
  { value: 'asgd', label: 'ASGD (Averaged Stochastic Gradient Descent)' },
  { value: 'lbfgs', label: 'L-BFGS' },
  { value: 'radam', label: 'RAdam' },
  { value: 'rmsprop', label: 'RMSprop' },
  { value: 'rprop', label: 'Rprop' },
  { value: 'sgd', label: 'SGD (Stochastic Gradient Descent)' }
];

export const lossFunctionOptions = [
  { value: '', label: 'Select a loss function...' },
  { value: 'ce', label: 'Cross Entropy' },
  { value: 'bce', label: 'Binary Cross Entropy' },
  { value: 'bcelogit', label: 'Binary Cross Entropy with Logits' },
  { value: 'mse', label: 'Mean Squared Error (MSE)' },
  { value: 'l1', label: 'L1 Loss (Mean Absolute Error)' },
  { value: 'nl', label: 'Negative Log Likelihood Loss' },
  { value: 'smoothl1', label: 'Smooth L1 Loss' },
  { value: 'kldiv', label: 'KL Divergence Loss' },
  { value: 'poissonnl', label: 'Poisson Negative Log Likelihood Loss' },
  { value: 'gaussiannl', label: 'Gaussian Negative Log Likelihood Loss' },
  { value: 'ctc', label: 'Connectionist Temporal Classification (CTC) Loss' },
  { value: 'huber', label: 'Huber Loss' },
  { value: 'softmargin', label: 'Soft Margin Loss' },
  { value: 'multilabelsoftmargin', label: 'Multi-Label Soft Margin Loss' },
  { value: 'cosineembedding', label: 'Cosine Embedding Loss' },
  { value: 'marginranking', label: 'Margin Ranking Loss' },
  { value: 'hingeembedding', label: 'Hinge Embedding Loss' }
];
