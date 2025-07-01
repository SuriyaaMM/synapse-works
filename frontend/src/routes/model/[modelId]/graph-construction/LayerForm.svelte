<script lang="ts">
import { createEventDispatcher } from 'svelte';
import type { LayerConfigInput, LinearLayerConfigInput, 
                Conv2dLayerConfigInput, Conv1dLayerConfigInput, 
                MaxPool2dLayerConfigInput, MaxPool1dLayerConfigInput,
                AvgPool2dLayerConfigInput, AvgPool1dLayerConfigInput, 
                BatchNorm2dLayerConfigInput, BatchNorm1dLayerConfigInput,
                FlattenLayerConfigInput, DropoutLayerConfigInput,
                ELULayerConfigInput, ReLULayerConfigInput,
                LeakyReLULayerConfigInput, SigmoidLayerConfigInput, 
                LogSigmoidLayerConfigInput, TanhLayerConfigInput} from '../../../../../../source/types/layerTypes';

const dispatch = createEventDispatcher();

export let selectedLayerType = null;
export let selectedNode = null;
export let selectedEdge = null;
export let nodes = [];
export let loading = false;
export let error = null;
export let buildResult = null;

// Form fields - Linear
  let layerName = '';
  let inFeatures = '';
  let outFeatures = '';
  let bias = true;
  
  // Form fields - Conv2d and Conv1d
  let inChannels = '';
  let outChannels = '';
  let kernelSize = '';
  let stride = '';
  let padding = '';
  let dilation = '';
  let groups = '1';
  let convBias = true;
  let paddingMode = 'zeros';

  // MaxPool2d and MaxPool1d layer fields
  let poolKernelSize = '';
  let poolStride = '';
  let poolPadding = '';
  let poolDilation = '';
  let returnIndices = false;
  let ceilMode = false;

  // AvgPool2d and AvgPool1d layer fields
  let countIncludePad = false;
  let divisorOverride = '';

  // BatchNorm layer fields
  let numFeatures = '';
  let eps = '1e-05';
  let momentum = '0.1';
  let affine = true;
  let trackRunningStatus = false;

  // Flatten layer fields
  let startDim = '1';
  let endDim = '-1';

  // Dropout layer fields
  let dropoutP = '0.5';

  // ELU layer fields
  let alpha = '1.0';
  let eluInplace = false;

  // ReLU layer fields
  let reluInplace = false;

  // LeakyReLU layer fields
  let negativeSlope = '0.01';
  let leakyReluInplace = false;

  function resetFormFields() {
    layerName = '';
    inFeatures = '';
    outFeatures = '';
    inChannels = '';
    outChannels = '';
    kernelSize = '';
    stride = '';
    padding = '';
    dilation = '';
    groups = '1';
    poolKernelSize = '';
    poolStride = '';
    poolPadding = '';
    poolDilation = '';
    returnIndices = false;
    ceilMode = false;
    countIncludePad = false;
    divisorOverride = '';
    numFeatures = '';
    eps = '';
    momentum = '0.1';
    affine = false;
    trackRunningStatus = false;
    startDim = '1';
    endDim = '-1';
    dropoutP = '0.5';
    alpha = '1.0';
    eluInplace = false;
    reluInplace = false;
    negativeSlope = '0.01';
    leakyReluInplace = false;
  }

  
  
  function populateFormFromNode(node: Node) {
    const config = node.layerConfig;
    
    if (config.type === 'linear' && config.linear) {
      layerName = config.linear.name ?? '';
      inFeatures = config.linear.in_features.toString();
      outFeatures = config.linear.out_features.toString();
      bias = config.linear.bias ?? true;
    } else if (config.type === 'conv2d' && config.conv2d) {
      layerName = config.conv2d.name ?? '';
      inChannels = config.conv2d.in_channels.toString();
      outChannels = config.conv2d.out_channels.toString();
      kernelSize = config.conv2d.kernel_size.join(',');
      stride = config.conv2d.stride ? config.conv2d.stride.join(',') : '';
      padding = config.conv2d.padding ? config.conv2d.padding.join(',') : '';
      dilation = config.conv2d.dilation ? config.conv2d.dilation.join(',') : '';
      groups = config.conv2d.groups ? config.conv2d.groups[0].toString() : '1';
      convBias = config.conv2d.bias ?? true;
      paddingMode = config.conv2d.padding_mode ?? 'zeros';
    } else if (config.type === 'conv1d' && config.conv1d) {
      layerName = config.conv1d.name ?? '';
      inChannels = config.conv1d.in_channels.toString();
      outChannels = config.conv1d.out_channels.toString();
      kernelSize = config.conv1d.kernel_size.join(',');
      stride = config.conv1d.stride ? config.conv1d.stride.join(',') : '';
      padding = config.conv1d.padding ? config.conv1d.padding.join(',') : '';
      dilation = config.conv1d.dilation ? config.conv1d.dilation.join(',') : '';
      groups = config.conv1d.groups ? config.conv1d.groups[0].toString() : '1';
      convBias = config.conv1d.bias ?? true;
      paddingMode = config.conv1d.padding_mode ?? 'zeros';
    } else if (config.type === 'maxpool2d' && config.maxpool2d) {
      layerName = config.maxpool2d.name ?? '';
      poolKernelSize = config.maxpool2d.kernel_size.join(',');
      poolStride = config.maxpool2d.stride ? config.maxpool2d.stride.join(',') : '';
      poolPadding = config.maxpool2d.padding ? config.maxpool2d.padding.join(',') : '';
      poolDilation = config.maxpool2d.dilation ? config.maxpool2d.dilation.join(',') : '';
      returnIndices = config.maxpool2d.return_indices ?? false;
      ceilMode = config.maxpool2d.ceil_mode ?? false;
    } else if (config.type === 'maxpool1d' && config.maxpool1d) {
      layerName = config.maxpool1d.name ?? '';
      poolKernelSize = config.maxpool1d.kernel_size.join(',');
      poolStride = config.maxpool1d.stride ? config.maxpool1d.stride.join(',') : '';
      poolPadding = config.maxpool1d.padding ? config.maxpool1d.padding.join(',') : '';
      poolDilation = config.maxpool1d.dilation ? config.maxpool1d.dilation.join(',') : '';
      returnIndices = config.maxpool1d.return_indices ?? false;
      ceilMode = config.maxpool1d.ceil_mode ?? false;
    } else if (config.type === 'avgpool2d' && config.avgpool2d) {
      layerName = config.avgpool2d.name ?? '';
      poolKernelSize = config.avgpool2d.kernel_size.join(',');
      poolStride = config.avgpool2d.stride ? config.avgpool2d.stride.join(',') : '';
      poolPadding = config.avgpool2d.padding ? config.avgpool2d.padding.join(',') : '';
      countIncludePad = config.avgpool2d.count_include_pad ?? false;
      divisorOverride = config.avgpool2d.divisor_override !== undefined && config.avgpool2d.divisor_override !== null
        ? config.avgpool2d.divisor_override.toString()
        : '';
    } else if (config.type === 'avgpool1d' && config.avgpool1d) {
      layerName = config.avgpool1d.name ?? '';
      poolKernelSize = config.avgpool1d.kernel_size.join(',');
      poolStride = config.avgpool1d.stride ? config.avgpool1d.stride.join(',') : '';
      poolPadding = config.avgpool1d.padding ? config.avgpool1d.padding.join(',') : '';
      countIncludePad = config.avgpool1d.count_include_pad ?? false;
      divisorOverride = config.avgpool1d.divisor_override !== undefined && config.avgpool1d.divisor_override !== null
        ? config.avgpool1d.divisor_override.toString()
        : '';
    } else if (config.type === 'batchnorm2d' && config.batchnorm2d) {
      layerName = config.batchnorm2d.name ?? '';
      numFeatures = config.batchnorm2d.num_features.toString();
      eps = config.batchnorm2d.eps?.toString() || '';
      momentum = config.batchnorm2d.momentum?.toString() || '0.1';
      affine = config.batchnorm2d.affine ?? true;
      trackRunningStatus = config.batchnorm2d.track_running_status ?? true;
    } else if (config.type === 'batchnorm1d' && config.batchnorm1d){
      layerName = config.batchnorm1d.name ?? '';
      numFeatures = config.batchnorm1d.num_features.toString();
      eps = config.batchnorm1d.eps?.toString() || '';
      momentum = config.batchnorm1d.momentum?.toString() || '0.1';
      affine = config.batchnorm1d.affine ?? true;
      trackRunningStatus = config.batchnorm1d.track_running_status ?? true;
    } else if (config.type === 'flatten' && config.flatten) {
      layerName = config.flatten.name ?? '';
      startDim = config.flatten.start_dim?.toString() || '1';
      endDim = config.flatten.end_dim?.toString() || '-1';
    } else if (config.type === 'dropout' && config.dropout) {
      layerName = config.dropout.name ?? '';
      dropoutP = config.dropout.p?.toString() || '0.5';
    } else if (config.type === 'elu' && config.elu) {
      layerName = config.elu.name ?? '';
      alpha = config.elu.alpha?.toString() || '1.0';
      eluInplace = config.elu.inplace ?? false;
    } else if (config.type === 'relu' && config.relu) {
      layerName = config.relu.name ?? '';
      reluInplace = config.relu.inplace ?? false;
    } else if (config.type === 'leakyrelu' && config.leakyrelu) {
      layerName = config.leakyrelu.name ?? '';
      negativeSlope = config.leakyrelu.negative_slope?.toString() || '0.01';
      leakyReluInplace = config.leakyrelu.inplace ?? false;
    } else if (config.type === 'sigmoid') {
      layerName = config.sigmoid?.name ?? '';
      // No additional fields for Sigmoid
    } else if (config.type === 'logsigmoid') {
      layerName = config.logsigmoid?.name ?? '';
      // No additional fields for LogSigmoid
    } else if (config.type === 'tanh') {
      layerName = config.tanh?.name ?? '';
      // No additional fields for Tanh
    } else {
      console.warn(`Unknown layer type: ${config.type}`);
    }
  }

function parseArrayInput(input: string): number[] | null {
    if (!input.trim()) return null;
    
    const parts = input.split(',').map(s => s.trim());
    const numbers = parts.map(p => Number(p));
    
    if (numbers.some(n => isNaN(n) || n <= 0)) return null;
    
    return numbers;
  }

  function validateForm(): string | null {
    if (!selectedLayerType) return 'Layer type is required';
    
    if (selectedLayerType.type === 'linear') {
      const inFeaturesNum = Number(inFeatures);
      const outFeaturesNum = Number(outFeatures);
      
      if (!inFeatures || isNaN(inFeaturesNum) || inFeaturesNum <= 0) {
        return 'Input features must be a positive number';
      }
      if (!outFeatures || isNaN(outFeaturesNum) || outFeaturesNum <= 0) {
        return 'Output features must be a positive number';
      }
    } else if (selectedLayerType.type === 'conv2d') {
      const inChannelsNum = Number(inChannels);
      const outChannelsNum = Number(outChannels);
      const groupsNum = Number(groups);
      
      if (!inChannels || isNaN(inChannelsNum) || inChannelsNum <= 0) {
        return 'Input channels must be a positive number';
      }
      if (!outChannels || isNaN(outChannelsNum) || outChannelsNum <= 0) {
        return 'Output channels must be a positive number';
      }
      if (!kernelSize.trim()) {
        return 'Kernel size is required';
      }
      if (!groups || isNaN(groupsNum) || groupsNum <= 0) {
        return 'Groups must be a positive number';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(kernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
        return 'Kernel size must be an array of exactly two numbers (e.g., "3,3")';
      }
    } else if (selectedLayerType.type === 'conv1d') {
      const inChannelsNum = Number(inChannels);
      const outChannelsNum = Number(outChannels);
      const groupsNum = Number(groups);
      
      if (!inChannels || isNaN(inChannelsNum) || inChannelsNum <= 0) {
        return 'Input channels must be a positive number';
      }
      if (!outChannels || isNaN(outChannelsNum) || outChannelsNum <= 0) {
        return 'Output channels must be a positive number';
      }
      if (!kernelSize.trim()) {
        return 'Kernel size is required';
      }
      if (!groups || isNaN(groupsNum) || groupsNum <= 0) {
        return 'Groups must be a positive number';
      }
      const kernelSizeArray = parseArrayInput(kernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
        return 'Kernel size must be a number (e.g., "3")';
      }
    } else if (selectedLayerType.type === 'maxpool2d') {
      if (!poolKernelSize.trim()) {
        return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
        return 'Kernel size must be an array of exactly two numbers (e.g., "2,2")';
      }
    } else if (selectedLayerType.type === 'maxpool1d') {
      if (!poolKernelSize.trim()) {
        return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
        return 'Kernel size must be a number (e.g., "2")';
      }
    } else if (selectedLayerType.type === 'avgpool2d') {
      if (!poolKernelSize.trim()) {
          return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
          return 'Kernel size must be an array of exactly two numbers (e.g., "2,2")';
      }
    } else if (selectedLayerType.type === 'avgpool1d') {
      if (!poolKernelSize.trim()) {
          return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
          return 'Kernel size must be a number (e.g., "2")';
      }
    } else if (selectedLayerType.type === 'batchnorm2d' || selectedLayerType.type === 'batchnorm1d') {
      const numFeaturesNum = Number(numFeatures);
      
      if (!numFeatures || isNaN(numFeaturesNum) || numFeaturesNum <= 0) {
          return 'Number of features must be a positive number';
      }
      
      const epsNum = Number(eps);
      if (isNaN(epsNum) || epsNum <= 0) {
          return 'Eps must be a positive number';
      }
      
      const momentumNum = Number(momentum);
      if (isNaN(momentumNum) || momentumNum < 0 || momentumNum > 1) {
          return 'Momentum must be between 0 and 1';
      }
    } else if (selectedLayerType.type === 'flatten') {
      const startDimNum = Number(startDim);
      const endDimNum = Number(endDim);
      
      if (isNaN(startDimNum)) {
        return 'Start dimension must be a valid number';
      }
      if (isNaN(endDimNum)) {
        return 'End dimension must be a valid number';
      }
    } else if (selectedLayerType.type === 'dropout') {
      const pNum = Number(dropoutP);
      
      if (isNaN(pNum) || pNum < 0 || pNum > 1) {
        return 'Dropout probability must be between 0 and 1';
      }
    } else if (selectedLayerType.type === 'elu') {
      const alphaNum = Number(alpha);
      
      if (isNaN(alphaNum) || alphaNum <= 0) {
        return 'Alpha must be a positive number';
      }
    } else if (selectedLayerType.type === 'leakyrelu') {
      const slopeNum = Number(negativeSlope);
      
      if (isNaN(slopeNum)) {
        return 'Negative slope must be a valid number';
      }
    } else if (selectedLayerType.type === 'sigmoid' || selectedLayerType.type === 'logsigmoid' || selectedLayerType.type === 'tanh') {
    // No additional validation needed for these layers
    }
    
    return null;
  }

  function createLayerConfig(): LayerConfigInput {
    const layerNameValue = layerName.trim() || `${selectedLayerType!.type}Layer${nodeCounter}`;
    
    if (selectedLayerType!.type === 'linear') {
      const linearConfig: LinearLayerConfigInput = {
        name: layerNameValue,
        in_features: parseInt(inFeatures, 10),
        out_features: parseInt(outFeatures, 10),
        bias
      };
      
      return {
        type: 'linear',
        linear: linearConfig
      };
    } else if (selectedLayerType!.type === 'conv2d') {
      const conv2dConfig: Conv2dLayerConfigInput = {
        name: layerNameValue,
        in_channels: parseInt(inChannels, 10),
        out_channels: parseInt(outChannels, 10),
        kernel_size: parseArrayInput(kernelSize) || [3, 3],
        bias: convBias,
        padding_mode: paddingMode
      };
      
      // Add optional fields if provided
      if (stride.trim()) {
        const strideArray = parseArrayInput(stride);
        if (strideArray) {
          conv2dConfig.stride = strideArray;
        }
      }
      if (padding.trim()) {
        const paddingArray = parseArrayInput(padding);
        if (paddingArray) {
          conv2dConfig.padding = paddingArray;
        }
      }
      if (dilation.trim()) {
        const dilationArray = parseArrayInput(dilation);
        if (dilationArray) {
          conv2dConfig.dilation = dilationArray;
        }
      }
      if (groups.trim() && groups !== '1') {
        conv2dConfig.groups = [parseInt(groups, 10)];
      }
      
      return {
        type: 'conv2d',
        conv2d: conv2dConfig
      };
    } else if (selectedLayerType!.type === 'conv1d') {
      const conv1dConfig: Conv1dLayerConfigInput = {
        name: layerNameValue,
        in_channels: parseInt(inChannels, 10),
        out_channels: parseInt(outChannels, 10),
        kernel_size: parseArrayInput(kernelSize) || [3],
        bias: convBias,
        padding_mode: paddingMode
      };
      
      // Add optional fields if provided
      if (stride.trim()) {
        const strideArray = parseArrayInput(stride);
        if (strideArray) {
          conv1dConfig.stride = strideArray;
        }
      }
      if (padding.trim()) {
        const paddingArray = parseArrayInput(padding);
        if (paddingArray) {
          conv1dConfig.padding = paddingArray;
        }
      }
      if (dilation.trim()) {
        const dilationArray = parseArrayInput(dilation);
        if (dilationArray) {
          conv1dConfig.dilation = dilationArray;
        }
      }
      if (groups.trim() && groups !== '1') {
        conv1dConfig.groups = [parseInt(groups, 10)];
      }
      
      return {
        type: 'conv1d',
        conv1d: conv1dConfig
      };
    } else if (selectedLayerType!.type === 'maxpool2d') {
      const maxpool2dConfig: MaxPool2dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2, 2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        dilation: poolDilation ? (parseArrayInput(poolDilation) ?? undefined) : undefined,
        return_indices: returnIndices,
        ceil_mode: ceilMode
      };
      
      return {
        type: 'maxpool2d',
        maxpool2d: maxpool2dConfig
      };
    } else if (selectedLayerType!.type === 'maxpool1d') {
      const maxpool1dConfig: MaxPool1dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        dilation: poolDilation ? (parseArrayInput(poolDilation) ?? undefined) : undefined,
        return_indices: returnIndices,
        ceil_mode: ceilMode
      };
      
      return {
        type: 'maxpool1d',
        maxpool1d: maxpool1dConfig
      };
    } else if (selectedLayerType!.type === 'avgpool2d') {
      const avgpool2dConfig: AvgPool2dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2, 2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        count_include_pad: countIncludePad,
        divisor_override: divisorOverride.trim() ? Number(divisorOverride) : undefined
      };
      
      return {
        type: 'avgpool2d',
        avgpool2d: avgpool2dConfig
      };
    } else if (selectedLayerType!.type === 'avgpool1d') {
      const avgpool1dConfig: AvgPool1dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        count_include_pad: countIncludePad,
        divisor_override: divisorOverride.trim() ? Number(divisorOverride) : undefined
      };

      return {
        type: 'avgpool1d',
        avgpool1d: avgpool1dConfig
      };
    } else if (selectedLayerType!.type === 'batchnorm2d') {
      const batchnorm2dConfig: BatchNorm2dLayerConfigInput = {
        name: layerNameValue,
        num_features: parseInt(numFeatures, 10),
        eps: parseFloat(eps) || 1e-5,
        momentum: parseFloat(momentum) || 0.1,
        affine,
        track_running_status: trackRunningStatus
      };
      
      return {
        type: 'batchnorm2d',
        batchnorm2d: batchnorm2dConfig
      };
    } else if (selectedLayerType!.type === 'batchnorm1d') {
      const batchnorm1dConfig: BatchNorm1dLayerConfigInput = {
        name: layerNameValue,
        num_features: parseInt(numFeatures, 10),
        eps: parseFloat(eps) || 1e-5,
        momentum: parseFloat(momentum) || 0.1,
        affine,
        track_running_status: trackRunningStatus
      };
      
      return {
        type: 'batchnorm1d',
        batchnorm1d: batchnorm1dConfig
      };
    } else if (selectedLayerType!.type === 'flatten') {
      const flattenConfig: FlattenLayerConfigInput = {
        name: layerNameValue,
        start_dim: parseInt(startDim, 10) || 1,
        end_dim: parseInt(endDim, 10) || -1
      };
      
      return {
        type: 'flatten',
        flatten: flattenConfig
      };
    } else if (selectedLayerType!.type === 'dropout') {
      const dropoutConfig: DropoutLayerConfigInput = {
        name: layerNameValue,
        p: parseFloat(dropoutP) || 0.5
      };
      
      return {
        type: 'dropout',
        dropout: dropoutConfig
      };
    } else if (selectedLayerType!.type === 'elu') {
      const eluConfig: ELULayerConfigInput = {
        name: layerNameValue,
        alpha: parseFloat(alpha) || 1.0,
        inplace: eluInplace
      };
      
      return {
        type: 'elu',
        elu: eluConfig
      };
    } else if (selectedLayerType!.type === 'relu') {
      const reluConfig: ReLULayerConfigInput = {
        name: layerNameValue,
        inplace: reluInplace
      };
      
      return {
        type: 'relu',
        relu: reluConfig
      };
    } else if (selectedLayerType!.type === 'leakyrelu') {
      const leakyReluConfig: LeakyReLULayerConfigInput = {
        name: layerNameValue,
        negative_slope: parseFloat(negativeSlope) || 0.01,
        inplace: leakyReluInplace
      };
      
      return {
        type: 'leakyrelu',
        leakyrelu: leakyReluConfig
      };
    } else if (selectedLayerType!.type === 'sigmoid') {
      const sigmoidConfig: SigmoidLayerConfigInput = {
        name: layerNameValue
      };
      return { 
        type: 'sigmoid', 
        sigmoid: sigmoidConfig
      };
    } else if (selectedLayerType!.type === 'logsigmoid') {
      const logsigmoidConfig: LogSigmoidLayerConfigInput = {
        name: layerNameValue
      };
      return { 
        type: 'logsigmoid', 
        logsigmoid: logsigmoidConfig
      };
    } else if (selectedLayerType!.type === 'tanh') {
      const tanhConfig: TanhLayerConfigInput = {
        name: layerNameValue
      };
      return { 
        type: 'tanh', 
        tanh: tanhConfig
      };
    }

    throw new Error(`Unsupported layer type: ${selectedLayerType!.type}`);
  }

  function handleAddLayer() {
  const validationError = validateForm();
  if (validationError) return;
  
  const layerConfig = createLayerConfig();
  dispatch('addLayer', { layerConfig });
  resetFormFields();
}

function handleDeleteNode() {
  if (!selectedNode) return;
  dispatch('deleteNode', { nodeId: selectedNode.id });
}

function handleDeleteEdge() {
  if (!selectedEdge) return;
  dispatch('deleteEdge', { edgeId: selectedEdge.id });
}

// Auto-populate form when node selection changes
$: if (selectedNode) {
  populateFormFromNode(selectedNode);
}

// Clear form when switching layer types
$: if (selectedLayerType) {
  resetFormFields();
}
