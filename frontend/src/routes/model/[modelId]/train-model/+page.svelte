<script lang="ts">
  import { page } from '$app/stores';
  import { onMount, onDestroy } from 'svelte';
  import type { TrainStatus } from '../../../../../../source/types/trainTypes';
  import { fetchModelDetails, modelDetails } from "../modelDetails";
  import { ExportType } from '../../../../../../source/types/argTypes';
  import { TrainingUtils } from './train-model-utils';

  import './train-model.css';
  
  let loading = false;
  let training = false;
  let error: string | null = null;
  let trainingStatus: TrainStatus | null = null;
  let stoppedByUser = false;
  let selectedExportType: ExportType = ExportType.ONNX;

  const trainingService = new TrainingUtils();

  fetchModelDetails();
  
  onMount(() => {
    trainingService.setCallbacks(
      handleStatusUpdate,
      handleTrainingComplete,
      handleError
    );
    
    // Check initial training status
    trainingService.checkTrainingStatus();
  });

  onDestroy(() => {
    trainingService.cleanup();
  });

  function handleStatusUpdate(status: TrainStatus | null) {
    trainingStatus = status;
    
    if (status && !status.completed) {
      training = true;
    }
  }

  function handleTrainingComplete() {
    training = false;
  }

  function handleError(errorMessage: string) {
    error = errorMessage;
    training = false;
    loading = false;
  }

  function stopTraining() {
    trainingService.stopTraining();
    training = false;
    stoppedByUser = true;
    
    if (trainingStatus) {
      trainingStatus = {
        ...trainingStatus,
        completed: true
      };
    }
  }

  async function startTraining() {
    // Validate model configuration
    const validationError = TrainingUtils.validateModelConfig($modelDetails);
    if (validationError) {
      error = validationError;
      return;
    }
    
    loading = true;
    training = true;
    error = null;
    stoppedByUser = false;

    const success = await trainingService.startTraining(selectedExportType);
    
    if (!success) {
      training = false;
    }
    
    loading = false;
  }
</script>

<div class="start-training-container">
  <h1 class="heading">Start Training</h1>
    <div class="space-y-6">
      {#if $modelDetails}
        <div class="model-overview">
          <h3 class="font-semibold model-info" style="margin-bottom: 12px; font-weight: 600;">Model Overview</h3>
          <div class="model-section">
            <div>
              <p class="model-info">
                <span>Model Name:</span> {$modelDetails.name}
              </p>
              <p class="model-info">
                Total Layers: <span class="font-semibold">{$modelDetails.module_graph?.layers?.length || 0}</span>
            Total Connections: <span class="font-semibold">{$modelDetails.module_graph?.edges?.length || 0}</span>
              </p>
            </div>

            {#if $modelDetails.train_config}
              <div>
                <p class="model-info">
                  <span>Epochs:</span> {$modelDetails.train_config.epochs}
                </p>
                <p class="model-info">
                  <span>Optimizer:</span> {$modelDetails.train_config.optimizer}
                </p>
                <p class="model-info">
                  <span>Learning Rate:</span> {$modelDetails.train_config.optimizer_config?.lr}
                </p>
                <p class="model-info">
                  <span>Loss Function:</span> {$modelDetails.train_config.loss_function}
                </p>
              </div>
            {/if}
          </div>

          {#if $modelDetails.dataset_config}
            <div class="dataset-info">
              <p>
                <span>Dataset:</span> {$modelDetails.dataset_config.name} 
                (Batch Size: {$modelDetails.dataset_config.batch_size}, 
                Split: {($modelDetails.dataset_config.split_length?.[1] != null ? ($modelDetails.dataset_config.split_length[1] * 100).toFixed(0) : 'N/A')}% test)
              </p>
            </div>
          {/if}
        </div>
      {/if}

      <!-- Configuration Status Check -->
      {#if $modelDetails}
        <div class="checklist-section">
          <h3 class="checklist-title">Pre-Training Checklist</h3>
          <div>
            <div class="checklist-item">
              <div class="check-circle {$modelDetails?.module_graph?.layers.length ? 'green' : 'red'}"></div>
              <span>Build Graph: {$modelDetails?.module_graph?.layers.length ? 'Ready' : 'Missing'}</span>
            </div>
            <div class="checklist-item">
              <div class="check-circle {$modelDetails.train_config ? 'green' : 'red'}"></div>
              <span>Training Configuration: {$modelDetails.train_config ? 'Ready' : 'Missing'}</span>
            </div>
            <div class="checklist-item">
              <div class="check-circle {$modelDetails.dataset_config ? 'green' : 'red'}"></div>
              <span>Dataset Configuration: {$modelDetails.dataset_config ? 'Ready' : 'Missing'}</span>
            </div>
          </div>
          
          {#if $modelDetails?.module_graph?.layers?.length && $modelDetails.train_config && $modelDetails.dataset_config}
            <div class="checklist-ready">
              ✅ All configurations complete. Ready to start training!
            </div>
          {:else}
            <div class="checklist-warning">
              ❌ Please complete all configurations before starting training.
            </div>
          {/if}
        </div>
      {/if}

      <!-- Export Type Selection -->
      <div class="export-section">
        <h3 class="export-title">Export Options</h3>
        <div class="export-dropdown">
          <label for="export-type">Export Type:</label>
          <select 
            id="export-type"
            bind:value={selectedExportType}
            disabled={training}
            class="export-select"
          >
            <option value="ONNX">ONNX (Default)</option>
            <option value="TorchTensor">TorchTensor</option>
          </select>
          <p class="export-description">
            The trained model will be exported in the selected format after training completes.
          </p>
        </div>
      </div>

      <!-- Training Status -->
      {#if trainingStatus}
        <div class="training-status">
          <h3 class="status-title">Training Status</h3>
          
          {#if trainingStatus.completed}
            <div class="training-completed">
              <p>✅ Training Completed!</p>
              <div style="margin-top: 8px; font-size: 14px;">
                <p>Final Epoch: {trainingStatus.epoch}</p>
                <p>Final Loss: {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                <p>Final Accuracy: {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                <p>Export Format: {selectedExportType}</p>
              </div>
            </div>
          {:else}
            <div class="training-in-progress">
              <p>🔄 Training in Progress...</p>
              <div class="status-grid">
                <div>
                  <p>Current Epoch: {trainingStatus.epoch || 0} / {$modelDetails?.train_config?.epochs || 'N/A'}</p>
                  <p>Status: Running</p>
                </div>
                <div>
                  <p>Current Loss: {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                  <p>Current Accuracy: {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                </div>
              </div>
              <p style="margin-top: 8px; font-size: 14px; color: #666;">
                Will export as: {selectedExportType}
              </p>
            </div>
          {/if}
        </div>
      {/if}
      
      <!-- Training Controls -->
      <div class="training-controls">
        {#if !training && !trainingStatus?.completed}
          <button 
            on:click={startTraining}
            disabled={loading || !$modelDetails?.module_graph?.layers?.length || !$modelDetails?.train_config || !$modelDetails?.dataset_config}
            class="start-button"
          >
            {loading ? 'Starting Training...' : '🚀 Start Training'}
          </button>
        {:else if training && !trainingStatus?.completed}
          <button 
            on:click={stopTraining}
            class="stop-button"
          >
            🛑 Stop Training
          </button>
          
          {#if stoppedByUser}
            <div class="training-error" style="margin-top: 16px;">
              <p>⚠️ Training stopped by user</p>
            </div>
          {/if}
        {/if}
      </div>

      {#if error}
        <div class="training-error" style="margin-top: 16px;">
          <h4>Training Error:</h4>
          <p>{error}</p>
        </div>
      {/if}
    </div>
</div>