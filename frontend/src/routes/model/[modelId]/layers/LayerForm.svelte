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
                LogSigmoidLayerConfigInput, TanhLayerConfigInput} from '../../../../../../source/types';

  export let loading = false;

  // Typed dispatcher
  const dispatch = createEventDispatcher<{
    submit: { layerConfig: LayerConfigInput };
    clear: undefined;
  }>();

  // Type-safe dispatch functions
  function dispatchSubmit(layerConfig: LayerConfigInput) {
    dispatch('submit', { layerConfig });
  }

  function dispatchClear() {
    dispatch('clear');
  }

  // Form fields
  let layerType = 'linear';
  let layerName = '';
  
  // Linear layer fields
  let inFeatures = '';
  let outFeatures = '';
  let bias = true;
  
  // Conv2d and Conv1d layer fields
  let inChannels = '';
  let outChannels = '';
  let kernelSize = '';
  let stride = '';
  let padding = '';
  let dilation = '';
  let groups = '1';
  let conv2dBias = true;
  let paddingMode = 'zeros';
  let conv1dBias = true;

  // MaxPool2d and MaxPool1d layer fields
  let poolKernelSize = '';
  let poolStride = '';
  let poolPadding = '';
  let poolDilation = '';
  let returnIndices = false;
  let ceilMode = false;

  // AvgPool2d and AvgPool1d layer fields
  let countIncludePad = true;
  let divisorOverride = '';

  // BatchNorm layer fields
  let numFeatures = '';
  let eps = '1e-05';
  let momentum = '0.1';
  let affine = true;
  let trackRunningStatus = true;

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

  /**
   * Validates the form fields based on layer type
   * @returns Error message or null if valid
   */
  function validateForm(): string | null {
    if (!layerType.trim()) return 'Layer type is required';
    
    if (layerType === 'linear') {
      const inFeaturesNum = Number(inFeatures);
      const outFeaturesNum = Number(outFeatures);
      
      if (!inFeatures || isNaN(inFeaturesNum) || inFeaturesNum <= 0) {
        return 'Input features must be a positive number';
      }
      if (!outFeatures || isNaN(outFeaturesNum) || outFeaturesNum <= 0) {
        return 'Output features must be a positive number';
      }
    } else if (layerType === 'conv2d') {
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
    } else if (layerType === 'conv1d') {
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
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
        return 'Kernel size must be a number (e.g., "3")';
      }
    } else if (layerType === 'maxpool2d') {
      if (!poolKernelSize.trim()) {
        return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
        return 'Kernel size must be an array of exactly two numbers (e.g., "2,2")';
      }
    } else if (layerType === 'maxpool1d') {
      if (!poolKernelSize.trim()) {
        return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
        return 'Kernel size must be a number (e.g., "2")';
      }
    } else if (layerType === 'avgpool2d') {
      if (!poolKernelSize.trim()) {
          return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
          return 'Kernel size must be an array of exactly two numbers (e.g., "2,2")';
      }
      } else if (layerType === 'avgpool1d') {
      if (!poolKernelSize.trim()) {
          return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
          return 'Kernel size must be a number (e.g., "2")';
      }
    } else if (layerType === 'batchnorm2d' || layerType === 'batchnorm1d') {
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
    } else if (layerType === 'flatten') {
      const startDimNum = Number(startDim);
      const endDimNum = Number(endDim);
      
      if (isNaN(startDimNum)) {
        return 'Start dimension must be a valid number';
      }
      if (isNaN(endDimNum)) {
        return 'End dimension must be a valid number';
      }
    } else if (layerType === 'dropout') {
      const pNum = Number(dropoutP);
      
      if (isNaN(pNum) || pNum < 0 || pNum > 1) {
        return 'Dropout probability must be between 0 and 1';
      }
    } else if (layerType === 'elu') {
      const alphaNum = Number(alpha);
      
      if (isNaN(alphaNum) || alphaNum <= 0) {
        return 'Alpha must be a positive number';
      }
    } else if (layerType === 'leakyrelu') {
      const slopeNum = Number(negativeSlope);
      
      if (isNaN(slopeNum)) {
        return 'Negative slope must be a valid number';
      }
    } else if (layerType === 'sigmoid' || layerType === 'logsigmoid' || layerType === 'tanh') {
    // No additional validation needed for these layers
    }
    
    return null;
  }

  function parseArrayInput(input: string): number[] | null {
    if (!input.trim()) return null;
    
    const parts = input.split(',').map(s => s.trim());
    const numbers = parts.map(p => Number(p));
    
    if (numbers.some(n => isNaN(n) || n <= 0)) return null;
    
    return numbers;
  }

  function createLayerConfig(): LayerConfigInput {
    const layerNameValue = layerName.trim() || `${layerType}Layer`;
    
    if (layerType === 'linear') {
      const linearConfig: LinearLayerConfigInput = {
        name: layerNameValue,
        in_features: parseInt(inFeatures, 10),
        out_features: parseInt(outFeatures, 10),
        bias
      };
      
      return {
        type: layerType.trim(),
        linear: linearConfig
      };
    } else if (layerType === 'conv2d') {
      const conv2dConfig: Conv2dLayerConfigInput = {
        name: layerNameValue,
        in_channels: parseInt(inChannels, 10),
        out_channels: parseInt(outChannels, 10),
        kernel_size: parseArrayInput(kernelSize) || [3],
        bias: conv2dBias,
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
        type: layerType.trim(),
        conv2d: conv2dConfig
      };
      
    } else if (layerType === 'conv1d') {
      const conv1dConfig: Conv1dLayerConfigInput = {
        name: layerNameValue,
        in_channels: parseInt(inChannels, 10),
        out_channels: parseInt(outChannels, 10),
        kernel_size: parseArrayInput(kernelSize) || [3],
        bias: conv1dBias,
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
        type: layerType.trim(),
        conv1d: conv1dConfig
      };
      
    } else if (layerType === 'maxpool2d') {
      const maxpool2dConfig: MaxPool2dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2, 2],
        return_indices: returnIndices,
        ceil_mode: ceilMode
      };
      
      // Add optional fields if provided
      if (poolStride.trim()) {
        const strideArray = parseArrayInput(poolStride);
        if (strideArray) {
          maxpool2dConfig.stride = strideArray;
        }
      }
      if (poolPadding.trim()) {
        const paddingArray = parseArrayInput(poolPadding);
        if (paddingArray) {
          maxpool2dConfig.padding = paddingArray;
        }
      }
      if (poolDilation.trim()) {
        const dilationArray = parseArrayInput(poolDilation);
        if (dilationArray) {
          maxpool2dConfig.dilation = dilationArray;
        }
      }
      
      return {
        type: layerType.trim(),
        maxpool2d: maxpool2dConfig
      };
      
    } else if (layerType === 'maxpool1d') {
      const maxpool1dConfig: MaxPool1dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2],
        return_indices: returnIndices,
        ceil_mode: ceilMode
      };
      
      // Add optional fields if provided
      if (poolStride.trim()) {
        const strideArray = parseArrayInput(poolStride);
        if (strideArray) {
          maxpool1dConfig.stride = strideArray;
        }
      }
      if (poolPadding.trim()) {
        const paddingArray = parseArrayInput(poolPadding);
        if (paddingArray) {
          maxpool1dConfig.padding = paddingArray;
        }
      }
      if (poolDilation.trim()) {
        const dilationArray = parseArrayInput(poolDilation);
        if (dilationArray) {
          maxpool1dConfig.dilation = dilationArray;
        }
      }
      
      return {
        type: layerType.trim(),
        maxpool1d: maxpool1dConfig
      };
    } else if (layerType === 'avgpool2d') {
        const avgpool2dConfig: AvgPool2dLayerConfigInput = {
            name: layerNameValue,
            kernel_size: parseArrayInput(poolKernelSize) || [2, 2],
            count_include_pad: countIncludePad,
            ceil_mode: ceilMode
        };
        
        // Add optional fields if provided
        if (poolStride.trim()) {
            const strideArray = parseArrayInput(poolStride);
            if (strideArray) {
                avgpool2dConfig.stride = strideArray;
            }
        }
        if (poolPadding.trim()) {
            const paddingArray = parseArrayInput(poolPadding);
            if (paddingArray) {
                avgpool2dConfig.padding = paddingArray;
            }
        }
        if (divisorOverride.trim()) {
            avgpool2dConfig.divisor_override = parseInt(divisorOverride, 10);
        }
        
        return {
            type: layerType.trim(),
            avgpool2d: avgpool2dConfig
        };
        
    } else if (layerType === 'avgpool1d') {
        const avgpool1dConfig: AvgPool1dLayerConfigInput = {
            name: layerNameValue,
            kernel_size: parseArrayInput(poolKernelSize) || [2],
            count_include_pad: countIncludePad,
            ceil_mode: ceilMode
        };
        
        // Add optional fields if provided
        if (poolStride.trim()) {
            const strideArray = parseArrayInput(poolStride);
            if (strideArray) {
                avgpool1dConfig.stride = strideArray;
            }
        }
        if (poolPadding.trim()) {
            const paddingArray = parseArrayInput(poolPadding);
            if (paddingArray) {
                avgpool1dConfig.padding = paddingArray;
            }
        }
        
        return {
            type: layerType.trim(),
            avgpool1d: avgpool1dConfig
        };
      
    } else if (layerType === 'batchnorm2d') {
        const batchnorm2dConfig: BatchNorm2dLayerConfigInput = {
          name: layerNameValue,
          num_features: parseInt(numFeatures, 10),
          eps: parseFloat(eps),
          momentum: parseFloat(momentum),
          affine: affine,
          track_running_status: trackRunningStatus
        };
        
        return {
          type: layerType.trim(),
          batchnorm2d: batchnorm2dConfig
        };
        
      } else if (layerType === 'batchnorm1d') {
        const batchnorm1dConfig: BatchNorm1dLayerConfigInput = {
          name: layerNameValue,
          num_features: parseInt(numFeatures, 10),
          eps: parseFloat(eps),
          momentum: parseFloat(momentum),
          affine: affine,
          track_running_status: trackRunningStatus
        };
  
        return {
          type: layerType.trim(),
          batchnorm1d: batchnorm1dConfig
        };
      } else if (layerType === 'flatten') {
          const flattenConfig: FlattenLayerConfigInput = {
              name: layerNameValue,
              start_dim: parseInt(startDim, 10),
              end_dim: parseInt(endDim, 10)
          };
          
          return {
              type: layerType.trim(),
              flatten: flattenConfig
          };
          
      } else if (layerType === 'dropout') {
          const dropoutConfig: DropoutLayerConfigInput = {
              name: layerNameValue,
              p: parseFloat(dropoutP)
          };
          
          return {
              type: layerType.trim(),
              dropout: dropoutConfig
          };
          
      } else if (layerType === 'elu') {
          const eluConfig: ELULayerConfigInput = {
              name: layerNameValue,
              alpha: parseFloat(alpha),
              inplace: eluInplace
          };
          
          return {
              type: layerType.trim(),
              elu: eluConfig
          };
          
      } else if (layerType === 'relu') {
          const reluConfig: ReLULayerConfigInput = {
              name: layerNameValue,
              inplace: reluInplace
          };
          
          return {
              type: layerType.trim(),
              relu: reluConfig
          };
          
      } else if (layerType === 'leakyrelu') {
          const leakyReluConfig: LeakyReLULayerConfigInput = {
              name: layerNameValue,
              negative_slope: parseFloat(negativeSlope),
              inplace: leakyReluInplace
          };
          
          return {
              type: layerType.trim(),
              leakyrelu: leakyReluConfig
          };
      } else if (layerType === 'sigmoid') {
        const sigmoidConfig: SigmoidLayerConfigInput = {
          name: layerNameValue
        };
        
        return {
          type: layerType.trim(),
          sigmoid: sigmoidConfig
        };
        
      } else if (layerType === 'logsigmoid') {
        const logsigmoidConfig: LogSigmoidLayerConfigInput = {
          name: layerNameValue
        };
        
        return {
          type: layerType.trim(),
          logsigmoid: logsigmoidConfig
        };
        
      } else if (layerType === 'tanh') {
        const tanhConfig: TanhLayerConfigInput = {
          name: layerNameValue
        };
        
        return {
          type: layerType.trim(),
          tanh: tanhConfig
        };
      }

    throw new Error(`Unsupported layer type: ${layerType}`);
  }

  function handleSubmit() {
    const validationError = validateForm();
    if (validationError) {
      dispatchClear();
      throw new Error(validationError);
    }
    
    const layerConfig = createLayerConfig();
    dispatchSubmit(layerConfig);
  }

  // Resets the form fields to their initial state
  export function resetForm() {
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
    countIncludePad = true;
    divisorOverride = '';
    numFeatures = '';
    eps = '1e-05';
    momentum = '0.1';
    affine = true;
    trackRunningStatus = true;
    startDim = '1';
    endDim = '-1';
    dropoutP = '0.5';
    alpha = '1.0';
    eluInplace = false;
    reluInplace = false;
    negativeSlope = '0.01';
    leakyReluInplace = false;
  }

  /**
   * Handles layer type change
   */
  function handleLayerTypeChange() {
    // Clear all fields when switching layer types
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
    countIncludePad = true;
    divisorOverride = '';
    numFeatures = '';
    eps = '1e-05';
    momentum = '0.1';
    affine = true;
    trackRunningStatus = true;
    startDim = '1';
    endDim = '-1';
    dropoutP = '0.5';
    alpha = '1.0';
    eluInplace = false;
    reluInplace = false;
    negativeSlope = '0.01';
    leakyReluInplace = false;
  }
</script>

<form on:submit|preventDefault={handleSubmit} class="space-y-2 max-w-md">
  <!-- Layer Type -->
  <div>
    <label for="layerType" class="block text-sm font-medium text-gray-700 mb-1">
      Layer Type <span class="text-red-500">*</span>
    </label>
    <select
      id="layerType"
      bind:value={layerType}
      on:change={handleLayerTypeChange}
      required
      class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      disabled={loading}
    >
    <option value="linear">Linear</option>
    <option value="conv2d">Conv2d</option>
    <option value="conv1d">Conv1d</option>
    <option value="maxpool2d">MaxPool2d</option>
    <option value="maxpool1d">MaxPool1d</option>
    <option value="avgpool2d">AvgPool2d</option>
    <option value="avgpool1d">AvgPool1d</option>
    <option value="batchnorm2d">BatchNorm2d</option>
    <option value="batchnorm1d">BatchNorm1d</option>
    <option value="flatten">Flatten</option>
    <option value="dropout">Dropout</option>
    <option value="elu">ELU</option>
    <option value="relu">ReLU</option>
    <option value="leakyrelu">LeakyReLU</option>
    <option value="sigmoid">Sigmoid</option>
    <option value="logsigmoid">LogSigmoid</option>
    <option value="tanh">Tanh</option>
    </select>
  </div>

  <!-- Layer Name -->
  <div>
    <label for="layerName" class="block text-sm font-medium text-gray-700 mb-1">
      Layer Name <span class="text-gray-400">(optional)</span>
    </label>
    <input
      id="layerName"
      type="text"
      bind:value={layerName}
      placeholder="e.g., InputLinear, HiddenLayer1, MaxPoolLayer"
      class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      disabled={loading}
    />
  </div>

  <!-- Linear Layer Fields -->
  {#if layerType === 'linear'}
    <!-- Input Features -->
    <div>
      <label for="inFeatures" class="block text-sm font-medium text-gray-700 mb-1">
        Input Features <span class="text-red-500">*</span>
      </label>
      <input
        id="inFeatures"
        type="number"
        bind:value={inFeatures}
        placeholder="e.g., 784"
        required
        min="1"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
    </div>

    <!-- Output Features -->
    <div>
      <label for="outFeatures" class="block text-sm font-medium text-gray-700 mb-1">
        Output Features <span class="text-red-500">*</span>
      </label>
      <input
        id="outFeatures"
        type="number"
        bind:value={outFeatures}
        placeholder="e.g., 64"
        required
        min="1"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
    </div>

    <!-- Bias -->
    <div class="flex items-center">
      <input
        id="bias"
        type="checkbox"
        bind:checked={bias}
        class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        disabled={loading}
      />
      <label for="bias" class="ml-2 block text-sm text-gray-700">
        Use bias
      </label>
    </div>
  {/if}

  <!-- Conv2d and Conv1d Layer Fields -->
  {#if layerType === 'conv2d' || layerType === 'conv1d'}
    <!-- Input Channels -->
    <div>
      <label for="inChannels" class="block text-sm font-medium text-gray-700 mb-1">
        Input Channels <span class="text-red-500">*</span>
      </label>
      <input
        id="inChannels"
        type="number"
        bind:value={inChannels}
        placeholder="e.g., 3"
        required
        min="1"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
    </div>

    <!-- Output Channels -->
    <div>
      <label for="outChannels" class="block text-sm font-medium text-gray-700 mb-1">
        Output Channels <span class="text-red-500">*</span>
      </label>
      <input
        id="outChannels"
        type="number"
        bind:value={outChannels}
        placeholder="e.g., 32"
        required
        min="1"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
    </div>

    <!-- Kernel Size -->
    <div>
      <label for="kernelSize" class="block text-sm font-medium text-gray-700 mb-1">
        Kernel Size <span class="text-red-500">*</span>
      </label>
      <input
        id="kernelSize"
        type="text"
        bind:value={kernelSize}
        placeholder={layerType === 'conv2d' ? "e.g., 3,3" : "e.g., 3"}
        required
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
      <p class="text-xs text-gray-500 mt-1">
        {layerType === 'conv2d' ? 'Format: "width,height" (e.g., "3,3")' : 'Format: single number (e.g., "3")'}
      </p>
    </div>

    <!-- Stride (Optional) -->
    <div>
      <label for="stride" class="block text-sm font-medium text-gray-700 mb-1">
        Stride <span class="text-gray-400">(optional)</span>
      </label>
      <input
        id="stride"
        type="text"
        bind:value={stride}
        placeholder={layerType === 'conv2d' ? "e.g., 1,1" : "e.g., 1"}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
      <p class="text-xs text-gray-500 mt-1">
        {layerType === 'conv2d' ? 'Format: "width,height" (e.g., "1,1")' : 'Format: single number (e.g., "1")'}
      </p>
    </div>

    <!-- Padding (Optional) -->
    <div>
      <label for="padding" class="block text-sm font-medium text-gray-700 mb-1">
        Padding <span class="text-gray-400">(optional)</span>
      </label>
      <input
        id="padding"
        type="text"
        bind:value={padding}
        placeholder={layerType === 'conv2d' ? "e.g., 1,1" : "e.g., 1"}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
      <p class="text-xs text-gray-500 mt-1">
        {layerType === 'conv2d' ? 'Format: "width,height" (e.g., "1,1")' : 'Format: single number (e.g., "1")'}
      </p>
    </div>

    <!-- Dilation (Optional) -->
    <div>
      <label for="dilation" class="block text-sm font-medium text-gray-700 mb-1">
        Dilation <span class="text-gray-400">(optional)</span>
      </label>
      <input
        id="dilation"
        type="text"
        bind:value={dilation}
        placeholder={layerType === 'conv2d' ? "e.g., 1,1" : "e.g., 1"}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
      <p class="text-xs text-gray-500 mt-1">
        {layerType === 'conv2d' ? 'Format: "width,height" (e.g., "1,1")' : 'Format: single number (e.g., "1")'}
      </p>
    </div>

    <!-- Groups -->
    <div>
      <label for="groups" class="block text-sm font-medium text-gray-700 mb-1">
        Groups
      </label>
      <input
        id="groups"
        type="number"
        bind:value={groups}
        placeholder="1"
        min="1"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
    </div>

    <!-- Padding Mode -->
    <div>
      <label for="paddingMode" class="block text-sm font-medium text-gray-700 mb-1">
        Padding Mode
      </label>
      <select
        id="paddingMode"
        bind:value={paddingMode}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      >
        <option value="zeros">Zeros</option>
        <option value="reflect">Reflect</option>
        <option value="replicate">Replicate</option>
        <option value="circular">Circular</option>
      </select>
    </div>

    <!-- Bias -->
    {#if layerType === 'conv2d'}
      <div class="flex items-center">
        <input
          id="conv2dBias"
          type="checkbox"
          bind:checked={conv2dBias}
          class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          disabled={loading}
        />
        <label for="conv2dBias" class="ml-2 block text-sm text-gray-700">
          Use bias
        </label>
      </div>
    {:else if layerType === 'conv1d'}
      <div class="flex items-center">
        <input
          id="conv1dBias"
          type="checkbox"
          bind:checked={conv1dBias}
          class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          disabled={loading}
        />
        <label for="conv1dBias" class="ml-2 block text-sm text-gray-700">
          Use bias
        </label>
      </div>
    {/if}
  {/if}

  <!-- MaxPool2d, MaxPool1d, AvgPool2d and AvgPool1d Layer Fields -->
  {#if layerType === 'maxpool2d' || layerType === 'maxpool1d' || layerType === 'avgpool2d' || layerType === 'avgpool1d'}
    <!-- Kernel Size -->
    <div>
      <label for="poolKernelSize" class="block text-sm font-medium text-gray-700 mb-1">
        Kernel Size <span class="text-red-500">*</span>
      </label>
      <input
        id="poolKernelSize"
        type="text"
        bind:value={poolKernelSize}
        placeholder={layerType === 'maxpool2d' || layerType === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"}
        required
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
      <p class="text-xs text-gray-500 mt-1">
        {(layerType === 'maxpool2d' || layerType === 'avgpool2d') ? 'Format: "width,height" (e.g., "2,2")' : 'Format: single number (e.g., "2")'}
      </p>
    </div>

    <!-- Stride (Optional) -->
    <div>
      <label for="poolStride" class="block text-sm font-medium text-gray-700 mb-1">
        Stride <span class="text-gray-400">(optional)</span>
      </label>
      <input
        id="poolStride"
        type="text"
        bind:value={poolStride}
        placeholder={layerType === 'maxpool2d' || layerType === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
      <p class="text-xs text-gray-500 mt-1">
        {(layerType === 'maxpool2d' || layerType === 'avgpool2d') ? 'Format: "width,height" (e.g., "2,2")' : 'Format: single number (e.g., "2")'}
      </p>
    </div>

    <!-- Padding (Optional) -->
    <div>
      <label for="poolPadding" class="block text-sm font-medium text-gray-700 mb-1">
        Padding <span class="text-gray-400">(optional)</span>
      </label>
      <input
        id="poolPadding"
        type="text"
        bind:value={poolPadding}
        placeholder={layerType === 'maxpool2d' || layerType === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
      <p class="text-xs text-gray-500 mt-1">
        {(layerType === 'maxpool2d' || layerType === 'avgpool2d') ? 'Format: "width,height" (e.g., "2,2")' : 'Format: single number (e.g., "2")'}
      </p>
    </div>

    <!-- Dilation (Optional - MaxPool only) -->
    {#if layerType === 'maxpool2d' || layerType === 'maxpool1d'}
    <div>
        <label for="poolDilation" class="block text-sm font-medium text-gray-700 mb-1">
        Dilation <span class="text-gray-400">(optional)</span>
        </label>
        <input
        id="poolDilation"
        type="text"
        bind:value={poolDilation}
        placeholder={layerType === 'maxpool2d' ? "e.g., 1,1" : "e.g., 1"}
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
        />
        <p class="text-xs text-gray-500 mt-1">
        {layerType === 'maxpool2d' ? 'Format: "width,height" (e.g., "1,1")' : 'Format: single number (e.g., "1")'}
        </p>
    </div>
    {/if}

    <!-- Return Indices (MaxPool only) -->
    {#if layerType === 'maxpool2d' || layerType === 'maxpool1d'}
    <div class="flex items-center">
        <input
        id="returnIndices"
        type="checkbox"
        bind:checked={returnIndices}
        class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        disabled={loading}
        />
        <label for="returnIndices" class="ml-2 block text-sm text-gray-700">
        Return indices
        </label>
    </div>
    {/if}

    <!-- Count Include Pad (AvgPool only) -->
    {#if layerType === 'avgpool2d' || layerType === 'avgpool1d'}
    <div class="flex items-center">
        <input
        id="countIncludePad"
        type="checkbox"
        bind:checked={countIncludePad}
        class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        disabled={loading}
        />
        <label for="countIncludePad" class="ml-2 block text-sm text-gray-700">
        Count include pad
        </label>
    </div>
    {/if}

    <!-- Divisor Override (AvgPool only) -->
    {#if layerType === 'avgpool2d' || layerType === 'avgpool1d'}
    <div>
        <label for="divisorOverride" class="block text-sm font-medium text-gray-700 mb-1">
        Divisor Override <span class="text-gray-400">(optional)</span>
        </label>
        <input
        id="divisorOverride"
        type="number"
        bind:value={divisorOverride}
        placeholder="e.g., 4"
        min="1"
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
        />
    </div>
    {/if}

    <!-- Ceil Mode -->
    <div class="flex items-center">
      <input
        id="ceilMode"
        type="checkbox"
        bind:checked={ceilMode}
        class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        disabled={loading}
      />
      <label for="ceilMode" class="ml-2 block text-sm text-gray-700">
        Ceil mode
      </label>
    </div>
  {/if}
   <!-- BatchNorm2d and BatchNorm1d Layer Fields -->
    {#if layerType === 'batchnorm2d' || layerType === 'batchnorm1d'}
      <!-- Number of Features -->
      <div>
          <label for="numFeatures" class="block text-sm font-medium text-gray-700 mb-1">
          Number of Features <span class="text-red-500">*</span>
          </label>
          <input
          id="numFeatures"
          type="number"
          bind:value={numFeatures}
          placeholder="e.g., 64"
          required
          min="1"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
          />
      </div>

      <!-- Eps -->
      <div>
          <label for="eps" class="block text-sm font-medium text-gray-700 mb-1">
          Eps <span class="text-gray-400">(optional)</span>
          </label>
          <input
          id="eps"
          type="text"
          bind:value={eps}
          placeholder="1e-05"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
          />
      </div>

      <!-- Momentum -->
      <div>
          <label for="momentum" class="block text-sm font-medium text-gray-700 mb-1">
          Momentum <span class="text-gray-400">(optional)</span>
          </label>
          <input
          id="momentum"
          type="text"
          bind:value={momentum}
          placeholder="0.1"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
          />
      </div>

      <!-- Affine -->
      <div class="flex items-center">
        <input
          id="affine"
          type="checkbox"
          bind:checked={affine}
          class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          disabled={loading}
        />
        <label for="affine" class="ml-2 block text-sm text-gray-700">
          Affine transformation
        </label>
      </div>

      <!-- Track Running Status -->
      <div class="flex items-center">
          <input
          id="trackRunningStatus"
          type="checkbox"
          bind:checked={trackRunningStatus}
          class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          disabled={loading}
          />
          <label for="trackRunningStatus" class="ml-2 block text-sm text-gray-700">
          Track running statistics
          </label>
      </div>
    {/if}
    <!-- Flatten Layer Fields -->
    {#if layerType === 'flatten'}
      <!-- Start Dimension -->
      <div>
        <label for="startDim" class="block text-sm font-medium text-gray-700 mb-1">
          Start Dimension <span class="text-gray-400">(optional)</span>
        </label>
        <input
          id="startDim"
          type="number"
          bind:value={startDim}
          placeholder="1"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
      </div>

      <!-- End Dimension -->
      <div>
        <label for="endDim" class="block text-sm font-medium text-gray-700 mb-1">
          End Dimension <span class="text-gray-400">(optional)</span>
        </label>
        <input
          id="endDim"
          type="number"
          bind:value={endDim}
          placeholder="-1"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
      </div>
    {/if}

    <!-- Dropout Layer Fields -->
    {#if layerType === 'dropout'}
      <!-- Dropout Probability -->
      <div>
        <label for="dropoutP" class="block text-sm font-medium text-gray-700 mb-1">
          Dropout Probability <span class="text-gray-400">(optional)</span>
        </label>
        <input
          id="dropoutP"
          type="text"
          bind:value={dropoutP}
          placeholder="0.5"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
        <p class="text-xs text-gray-500 mt-1">Value between 0 and 1</p>
      </div>
    {/if}

    <!-- ELU Layer Fields -->
    {#if layerType === 'elu'}
      <!-- Alpha -->
      <div>
        <label for="alpha" class="block text-sm font-medium text-gray-700 mb-1">
          Alpha <span class="text-gray-400">(optional)</span>
        </label>
        <input
          id="alpha"
          type="text"
          bind:value={alpha}
          placeholder="1.0"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
      </div>

      <!-- Inplace -->
      <div class="flex items-center">
        <input
          id="eluInplace"
          type="checkbox"
          bind:checked={eluInplace}
          class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          disabled={loading}
        />
        <label for="eluInplace" class="ml-2 block text-sm text-gray-700">
          In-place operation
        </label>
      </div>
    {/if}

    <!-- ReLU Layer Fields -->
    {#if layerType === 'relu'}
      <!-- Inplace -->
      <div class="flex items-center">
        <input
          id="reluInplace"
          type="checkbox"
          bind:checked={reluInplace}
          class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          disabled={loading}
        />
        <label for="reluInplace" class="ml-2 block text-sm text-gray-700">
          In-place operation
        </label>
      </div>
    {/if}

    <!-- LeakyReLU Layer Fields -->
    {#if layerType === 'leakyrelu'}
      <!-- Negative Slope -->
      <div>
        <label for="negativeSlope" class="block text-sm font-medium text-gray-700 mb-1">
          Negative Slope <span class="text-gray-400">(optional)</span>
        </label>
        <input
          id="negativeSlope"
          type="text"
          bind:value={negativeSlope}
          placeholder="0.01"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
      </div>

      <!-- Inplace -->
      <div class="flex items-center">
        <input
          id="leakyReluInplace"
          type="checkbox"
          bind:checked={leakyReluInplace}
          class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          disabled={loading}
        />
        <label for="leakyReluInplace" class="ml-2 block text-sm text-gray-700">
          In-place operation
        </label>
      </div>
    {/if}

  <!-- Submit Button -->
  <button 
    type="submit"
    disabled={loading}
    class="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
  >
    {loading ? 'Adding Layer...' : 'Add Layer'}
  </button>
</form>