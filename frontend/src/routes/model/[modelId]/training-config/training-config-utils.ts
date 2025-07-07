import { optimizerConfigs, type OptimizerSchema } from './optimizer-config';
import type { OptimizerConfig, TrainConfig } from '../../../../../../source/types/trainTypes';
import client from '$lib/apolloClient';
import { SET_TRAIN_CONFIG } from '$lib/mutations';

export function initializeOptimizerConfig(optimizer: string): OptimizerConfig {
  const config = optimizerConfigs[optimizer as keyof typeof optimizerConfigs];
  if (!config) {
    return { lr: 0.001 };
  }
  
  // Initialize with learning rate (required) and clear other optional fields
  return { lr: config.lr?.default ?? 0.001 };
}

export function validateForm(
  epochs: number | null,
  optimizer: string,
  lossFunction: string,
  optimizerConfig: OptimizerConfig
): string | null {
  // Validate basic required fields
  if (!epochs || epochs <= 0) return 'Epochs must be a positive number';
  if (!optimizer.trim()) return 'Optimizer is required';
  if (!lossFunction.trim()) return 'Loss function is required';
  
  // Validate learning rate (always required)
  const lr = (optimizerConfig as any).lr;
  if (lr === undefined || lr === null || lr === '' || isNaN(Number(lr))) {
    return 'Learning rate is required and must be a valid number';
  }
  
  // Validate optimizer-specific parameters
  const config = optimizerConfigs[optimizer as keyof typeof optimizerConfigs];
  if (!config) return null;
  
  for (const [key, paramConfig] of Object.entries(config)) {
    const value = (optimizerConfig as any)[key];
    
    // Skip validation for optional fields that are empty/null/undefined
    if (!paramConfig.required && (value === undefined || value === null || value === '')) {
      continue;
    }
    
    // For required fields or provided optional fields, validate
    const finalValue = value !== undefined && value !== null && value !== '' ? value : paramConfig.default;
    
    const validationError = validateParameterValue(key, finalValue, paramConfig);
    if (validationError) return validationError;
  }
  
  return null;
}

function validateParameterValue(key: string, value: any, paramConfig: any): string | null {
  if (paramConfig.type === 'number') {
    if (typeof value !== 'number' || isNaN(value)) {
      return `${paramConfig.label} must be a valid number`;
    }
    if (value < paramConfig.min || value > paramConfig.max) {
      return `${paramConfig.label} must be between ${paramConfig.min} and ${paramConfig.max}`;
    }
  } else if (paramConfig.type === 'array') {
    if (!Array.isArray(value)) {
      return `${paramConfig.label} must be an array`;
    }
    if (paramConfig.subtype === 'number') {
      for (let i = 0; i < value.length; i++) {
        if (typeof value[i] !== 'number' || isNaN(value[i])) {
          return `${paramConfig.label}[${i}] must be a valid number`;
        }
        if (value[i] < paramConfig.min || value[i] > paramConfig.max) {
          return `${paramConfig.label}[${i}] must be between ${paramConfig.min} and ${paramConfig.max}`;
        }
      }
    }
  }
  
  return null;
}

export function prepareFinalOptimizerConfig(
  optimizer: string,
  optimizerConfig: OptimizerConfig
): OptimizerConfig {
  const finalConfig: any = {};
  const config = optimizerConfigs[optimizer as keyof typeof optimizerConfigs];
  
  if (!config) return optimizerConfig;
  
  Object.entries(config).forEach(([key, paramConfig]) => {
    const value = (optimizerConfig as any)[key];
    
    // For required fields, always include (use default if empty)
    if (paramConfig.required) {
      finalConfig[key] = value !== undefined && value !== null && value !== '' ? value : paramConfig.default;
    } 
    // For optional fields, only include if value is provided
    else if (value !== undefined && value !== null && value !== '') {
      finalConfig[key] = value;
    }
    // If no value provided for optional field, use default
    else {
      finalConfig[key] = paramConfig.default;
    }
  });
  
  return finalConfig;
}

export function formatScientificNumber(value: number): string {
  return value.toExponential(2);
}

export function updateArrayValue(
  optimizerConfig: OptimizerConfig,
  key: string,
  index: number,
  value: string | number,
  optimizer: string
): OptimizerConfig {
  const config = optimizerConfigs[optimizer][key];
  const currentArray = (optimizerConfig as any)[key] || [...config.default];
  const numValue = typeof value === 'number' ? value : parseFloat(value);
  
  if (!isNaN(numValue)) {
    currentArray[index] = numValue;
    return { ...optimizerConfig, [key]: [...currentArray] };
  }
  
  return optimizerConfig;
}

export interface TrainingConfigRequest {
  epochs: number;
  optimizer: string;
  optimizerConfig: OptimizerConfig;
  lossFunction: string;
  lossFunctionConfig?: {
    reduction?: string;
    ignore_index?: number | null;
    label_smoothing?: number;
  };
  metrics: {
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
    accuracy_visualization: boolean;
    loss_visualization: boolean;
    test_validation: boolean;
    test_validation_period: number;
    train_validation: boolean;
    train_validation_period: number;
  };
}

export interface ServiceResponse<T> {
  data?: T;
  error?: string;
}

export async function setTrainingConfig(
  request: TrainingConfigRequest
): Promise<ServiceResponse<TrainConfig>> {
  try {
    const response = await client.mutate({
    mutation: SET_TRAIN_CONFIG,
    variables: {
      epochs: request.epochs || 10,
      optimizer: request.optimizer,
      optimizerConfig: request.optimizerConfig,
      loss_function: request.lossFunction,
      loss_function_config: request.lossFunctionConfig,
      metrics: request.metrics
    }
  });

    console.log('Set training config response:', response);

    if (!response.data?.setTrainConfig) {
      throw new Error('Failed to set training configuration - no data returned');
    }
    
    return { data: response.data.setTrainConfig };
  } catch (err: any) {
    console.error('Apollo Error:', err);
    const errorMessage = err.message || err.toString() || 'Unknown error occurred';
    return { error: errorMessage };
  }
}