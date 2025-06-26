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

  import './layer-form.css';

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

<form on:submit|preventDefault={handleSubmit} class="form-container">
  <!-- Layer Type -->
  <div>
    <label for="layerType">
      Layer Type <span class="required">*</span>
    </label>
    <select
      id="layerType"
      bind:value={layerType}
      on:change={handleLayerTypeChange}
      required
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
    <label for="layerName">Layer Name <span class="optional">(optional)</span></label>
    <input id="layerName" type="text" bind:value={layerName} placeholder="e.g., InputLinear, HiddenLayer1, MaxPoolLayer" disabled={loading} />
  </div>

  <!-- Conditional Sections -->
  {#if layerType === 'linear'}
    <div>
      <label for="inFeatures">Input Features <span class="required">*</span></label>
      <input id="inFeatures" type="number" bind:value={inFeatures} placeholder="e.g., 784" required min="1" disabled={loading} />
    </div>
    <div>
      <label for="outFeatures">Output Features <span class="required">*</span></label>
      <input id="outFeatures" type="number" bind:value={outFeatures} placeholder="e.g., 64" required min="1" disabled={loading} />
    </div>
    <div class="checkbox-container">
      <input id="bias" type="checkbox" bind:checked={bias} disabled={loading} />
      <label for="bias">Use bias</label>
    </div>
  {/if}

  {#if layerType === 'conv2d' || layerType === 'conv1d'}
    <div>
      <label for="inChannels">Input Channels <span class="required">*</span></label>
      <input id="inChannels" type="number" bind:value={inChannels} placeholder="e.g., 3" required min="1" disabled={loading} />
    </div>
    <div>
      <label for="outChannels">Output Channels <span class="required">*</span></label>
      <input id="outChannels" type="number" bind:value={outChannels} placeholder="e.g., 32" required min="1" disabled={loading} />
    </div>
    <div>
      <label for="kernelSize">Kernel Size <span class="required">*</span></label>
      <input id="kernelSize" type="text" bind:value={kernelSize} placeholder={layerType === 'conv2d' ? "e.g., 3,3" : "e.g., 3"} required disabled={loading} />
      <p class="input-note">{layerType === 'conv2d' ? 'Format: "width,height" (e.g., "3,3")' : 'Format: single number (e.g., "3")'}</p>
    </div>
    <div>
      <label for="stride">Stride <span class="optional">(optional)</span></label>
      <input id="stride" type="text" bind:value={stride} placeholder={layerType === 'conv2d' ? "e.g., 1,1" : "e.g., 1"} disabled={loading} />
      <p class="input-note">{layerType === 'conv2d' ? 'Format: "width,height" (e.g., "1,1")' : 'Format: single number (e.g., "1")'}</p>
    </div>
    <div>
      <label for="padding">Padding <span class="optional">(optional)</span></label>
      <input id="padding" type="text" bind:value={padding} placeholder={layerType === 'conv2d' ? "e.g., 0,0" : "e.g., 0"} disabled={loading} />
      <p class="input-note">{layerType === 'conv2d' ? 'Format: "width,height" (e.g., "0,0")' : 'Format: single number (e.g., "0")'}</p>
    </div>
    <div>
      <label for="dilation">Dilation <span class="optional">(optional)</span></label>
      <input id="dilation" type="text" bind:value={dilation} placeholder={layerType === 'conv2d' ? "e.g., 1,1" : "e.g., 1"} disabled={loading} />
      <p class="input-note">{layerType === 'conv2d' ? 'Format: "width,height" (e.g., "1,1")' : 'Format: single number (e.g., "1")'}</p>
    </div>
    <div>
      <label for="groups">Groups</label>
      <input id="groups" type="number" bind:value={groups} placeholder="1" min="1" disabled={loading} />
    </div>
    <div>
      <label for="paddingMode">Padding Mode</label>
      <select id="paddingMode" bind:value={paddingMode} disabled={loading}>
        <option value="zeros">Zeros</option>
        <option value="reflect">Reflect</option>
        <option value="replicate">Replicate</option>
        <option value="circular">Circular</option>
      </select>
    </div>
    {#if layerType === 'conv2d'}
      <div class="checkbox-container">
        <input id="conv2dBias" type="checkbox" bind:checked={conv2dBias} disabled={loading} />
        <label for="conv2dBias">Use bias</label>
      </div>
    {:else if layerType === 'conv1d'}
      <div class="checkbox-container">
        <input id="conv1dBias" type="checkbox" bind:checked={conv1dBias} disabled={loading} />
        <label for="conv1dBias">Use bias</label>
      </div>
    {/if}
  {/if}

  <!-- MaxPool2d, MaxPool1d, AvgPool2d, AvgPool1d -->
  {#if layerType === 'maxpool2d' || layerType === 'maxpool1d' || layerType === 'avgpool2d' || layerType === 'avgpool1d'}
    <!-- Kernel Size -->
    <div>
      <label for="poolKernelSize">Kernel Size <span class="required">*</span></label>
      <input id="poolKernelSize" type="text" bind:value={poolKernelSize} required disabled={loading} placeholder={layerType === 'maxpool2d' || layerType === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"} />
      <p class="input-note">{(layerType === 'maxpool2d' || layerType === 'avgpool2d') ? 'Format: "width,height" (e.g., "2,2")' : 'Format: single number (e.g., "2")'}</p>
    </div>

    <!-- Stride -->
    <div>
      <label for="poolStride">Stride <span class="optional">(optional)</span></label>
      <input id="poolStride" type="text" bind:value={poolStride} disabled={loading} placeholder={layerType === 'maxpool2d' || layerType === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"} />
      <p class="input-note">{(layerType === 'maxpool2d' || layerType === 'avgpool2d') ? 'Format: "width,height" (e.g., "2,2")' : 'Format: single number (e.g., "2")'}</p>
    </div>

    <!-- Padding -->
    <div>
      <label for="poolPadding">Padding <span class="optional">(optional)</span></label>
      <input id="poolPadding" type="text" bind:value={poolPadding} disabled={loading} placeholder={layerType === 'maxpool2d' || layerType === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"} />
      <p class="input-note">{(layerType === 'maxpool2d' || layerType === 'avgpool2d') ? 'Format: "width,height" (e.g., "2,2")' : 'Format: single number (e.g., "2")'}</p>
    </div>

    <!-- Dilation -->
    {#if layerType === 'maxpool2d' || layerType === 'maxpool1d'}
      <div>
        <label for="poolDilation">Dilation <span class="optional">(optional)</span></label>
        <input id="poolDilation" type="text" bind:value={poolDilation} disabled={loading} placeholder={layerType === 'maxpool2d' ? "e.g., 1,1" : "e.g., 1"} />
        <p class="input-note">{layerType === 'maxpool2d' ? 'Format: "width,height" (e.g., "1,1")' : 'Format: single number (e.g., "1")'}</p>
      </div>

      <!-- Return Indices -->
      <div class="checkbox-container">
        <input id="returnIndices" type="checkbox" bind:checked={returnIndices} disabled={loading} />
        <label for="returnIndices">Return indices</label>
      </div>
    {/if}

    <!-- Count Include Pad -->
    {#if layerType === 'avgpool2d' || layerType === 'avgpool1d'}
      <div class="checkbox-container">
        <input id="countIncludePad" type="checkbox" bind:checked={countIncludePad} disabled={loading} />
        <label for="countIncludePad">Count include pad</label>
      </div>

      <!-- Divisor Override -->
      <div>
        <label for="divisorOverride">Divisor Override <span class="optional">(optional)</span></label>
        <input id="divisorOverride" type="number" bind:value={divisorOverride} min="1" placeholder="e.g., 4" disabled={loading} />
      </div>
    {/if}

    <!-- Ceil Mode -->
    <div class="checkbox-container">
      <input id="ceilMode" type="checkbox" bind:checked={ceilMode} disabled={loading} />
      <label for="ceilMode">Ceil mode</label>
    </div>
  {/if}

  <!-- BatchNorm Layers -->
  {#if layerType === 'batchnorm2d' || layerType === 'batchnorm1d'}
    <div>
      <label for="numFeatures">Number of Features <span class="required">*</span></label>
      <input id="numFeatures" type="number" bind:value={numFeatures} required min="1" placeholder="e.g., 64" disabled={loading} />
    </div>

    <div>
      <label for="eps">Eps <span class="optional">(optional)</span></label>
      <input id="eps" type="text" bind:value={eps} placeholder="1e-05" disabled={loading} />
    </div>

    <div>
      <label for="momentum">Momentum <span class="optional">(optional)</span></label>
      <input id="momentum" type="text" bind:value={momentum} placeholder="0.1" disabled={loading} />
    </div>

    <div class="checkbox-container">
      <input id="affine" type="checkbox" bind:checked={affine} disabled={loading} />
      <label for="affine">Affine transformation</label>
    </div>

    <div class="checkbox-container">
      <input id="trackRunningStatus" type="checkbox" bind:checked={trackRunningStatus} disabled={loading} />
      <label for="trackRunningStatus">Track running statistics</label>
    </div>
  {/if}

  <!-- Flatten Layer -->
  {#if layerType === 'flatten'}
    <div>
      <label for="startDim">Start Dimension <span class="optional">(optional)</span></label>
      <input id="startDim" type="number" bind:value={startDim} placeholder="1" disabled={loading} />
    </div>

    <div>
      <label for="endDim">End Dimension <span class="optional">(optional)</span></label>
      <input id="endDim" type="number" bind:value={endDim} placeholder="-1" disabled={loading} />
    </div>
  {/if}

  <!-- Dropout Layer -->
  {#if layerType === 'dropout'}
    <div>
      <label for="dropoutP">Dropout Probability <span class="optional">(optional)</span></label>
      <input id="dropoutP" type="text" bind:value={dropoutP} placeholder="0.5" disabled={loading} />
      <p class="input-note">Value between 0 and 1</p>
    </div>
  {/if}

  <!-- ELU Layer -->
  {#if layerType === 'elu'}
    <div>
      <label for="alpha">Alpha <span class="optional">(optional)</span></label>
      <input id="alpha" type="text" bind:value={alpha} placeholder="1.0" disabled={loading} />
    </div>

    <div class="checkbox-container">
      <input id="eluInplace" type="checkbox" bind:checked={eluInplace} disabled={loading} />
      <label for="eluInplace">In-place operation</label>
    </div>
  {/if}

  <!-- ReLU Layer -->
  {#if layerType === 'relu'}
    <div class="checkbox-container">
      <input id="reluInplace" type="checkbox" bind:checked={reluInplace} disabled={loading} />
      <label for="reluInplace">In-place operation</label>
    </div>
  {/if}

  <!-- LeakyReLU Layer -->
  {#if layerType === 'leakyrelu'}
    <div>
      <label for="negativeSlope">Negative Slope <span class="optional">(optional)</span></label>
      <input id="negativeSlope" type="text" bind:value={negativeSlope} placeholder="0.01" disabled={loading} />
    </div>

    <div class="checkbox-container">
      <input id="leakyReluInplace" type="checkbox" bind:checked={leakyReluInplace} disabled={loading} />
      <label for="leakyReluInplace">In-place operation</label>
    </div>
  {/if}

  <!-- Submit Button -->
  <button type="submit" disabled={loading} class="submit-button">
    {loading ? 'Adding Layer...' : 'Add Layer'}
  </button>
</form>