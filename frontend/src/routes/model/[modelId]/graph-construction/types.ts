import type { LayerConfigInput } from "../../../../../../source/types/layerTypes";

export interface FlowNode {
  id: string;
  type?: string;
  position: { x: number; y: number };
  data: {
    label: string;
    layerType: string;
    layerConfig: LayerConfigInput;
    color: string;
  };
  selected?: boolean;
  selectable?: boolean;
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
}

export const layerTypes = [
  { type: 'linear', color: '#DC2626', label: 'Linear' },
  { type: 'conv2d', color: '#4F46E5', label: 'Conv2D' },
  { type: 'conv1d', color: '#7C3AED', label: 'Conv1D' },
  { type: 'convtranspose2d', color: '#E11D48', label: 'ConvTranspose2D' },
  { type: 'maxpool2d', color: '#0D9488', label: 'MaxPool2D' },
  { type: 'maxpool1d', color: '#14B8A6', label: 'MaxPool1D' },
  { type: 'avgpool2d', color: '#0EA5E9', label: 'AvgPool2D' },
  { type: 'avgpool1d', color: '#38BDF8', label: 'AvgPool1D' },
  { type: 'batchnorm2d', color: '#F59E0B', label: 'BatchNorm2D' },
  { type: 'batchnorm1d', color: '#FBBF24', label: 'BatchNorm1D' },
  { type: 'flatten', color: '#6B7280', label: 'Flatten' },
  { type: 'dropout', color: '#9CA3AF', label: 'Dropout' },
  { type: 'dropout2d', color: '#D1D5DB', label: 'Dropout2D' },
  { type: 'elu', color: '#9333EA', label: 'ELU' },
  { type: 'relu', color: '#10B981', label: 'ReLU' },
  { type: 'leakyrelu', color: '#8B5CF6', label: 'LeakyReLU' },
  { type: 'sigmoid', color: '#EF4444', label: 'Sigmoid' },
  { type: 'logsigmoid', color: '#F87171', label: 'LogSigmoid' },
  { type: 'tanh', color: '#6366F1', label: 'Tanh' },
  { type: 'cat', color: '#FB923C', label: 'Cat' }
];

export interface GraphState {
  nodes: FlowNode[];
  edges: FlowEdge[];
  selectedNode: FlowNode | null;
  selectedEdge: FlowEdge | null;
  selectedLayerType: typeof layerTypes[number] | null;
  nodeCounter: number;
  buildResult: any;
  graphValidationResult: any;
  loading: boolean;
  error: string | null;
}