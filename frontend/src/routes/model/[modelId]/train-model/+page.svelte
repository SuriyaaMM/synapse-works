<script lang="ts">
  import { page } from '$app/stores';
  import { onMount, onDestroy } from 'svelte';
  import client from '$lib/apolloClient';
  import { TRAIN_MODEL } from '$lib/mutations';
  import { GET_MODEL, GET_TRAINING_STATUS } from '$lib/queries';
  import type { Model, TrainStatus } from '../../../../../../source/types';

  import './train-model.css';
  
  let modelId: string | null = null;
  let loading = false;
  let training = false;
  let error: string | null = null;
  let modelDetails: Model | null = null;
  let trainingStatus: TrainStatus | null = null;
  let statusInterval: any = null;
  let stoppedByUser = false;

  // Extract modelId from URL path instead of query parameters
  $: {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    if (modelIndex !== -1 && modelIndex + 1 < pathParts.length) {
      modelId = pathParts[modelIndex + 1];
    } else {
      modelId = null;
    }
  }

  // Fetch model details when modelId changes
  $: if (modelId) {
    fetchModelDetails();
  }

  onMount(() => {
    // Check if training is already in progress
    if (modelId) {
      checkTrainingStatus();
    }
  });

  onDestroy(() => {
    if (statusInterval) {
      clearInterval(statusInterval);
    }
  });

  async function fetchModelDetails() {
    if (!modelId) return;
    
    try {
      const response = await client.query({
        query: GET_MODEL,
        variables: { id: modelId },
        fetchPolicy: 'network-only'
      });
      
      modelDetails = response.data?.getModel;
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = 'Failed to fetch model details';
    }
  }

  async function checkTrainingStatus() {
    if (!modelId) return;
    
    try {
      const response = await client.query({
        query: GET_TRAINING_STATUS,
        variables: { modelId },
        fetchPolicy: 'no-cache' // Force fresh data
      });
      
      const newStatus = response.data?.getTrainingStatus;
      
      if (newStatus) {
        trainingStatus = newStatus;
        
        // If training is in progress, start polling
        if (!newStatus.completed && !statusInterval) {
          training = true;
          startStatusPolling();
        }
      }
    } catch (err) {
      console.error('Error checking training status:', err);
    }
  }

  function stopTraining() {
    // Stop polling
    if (statusInterval) {
      clearInterval(statusInterval);
      statusInterval = null;
    }
    
    // Reset training state
    training = false;
    stoppedByUser = true;  // Add this line
    
    // Reset training status
    if (trainingStatus) {
      trainingStatus = {
        ...trainingStatus,
        completed: true
      };
    }
    
    console.log('Training stopped by user');
  }

  function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    
    statusInterval = setInterval(async () => {
      try {
        const response = await client.query({
          query: GET_TRAINING_STATUS,
          variables: { modelId },
          fetchPolicy: 'no-cache' // Always get fresh data
        });
        
        const newStatus = response.data?.getTrainingStatus;
        
        // Always update if we get new data
        if (newStatus) {
          trainingStatus = newStatus;
          
          // Stop polling if training is completed
          if (newStatus.completed) {
            clearInterval(statusInterval);
            statusInterval = null;
            training = false;
          }
        }
        
      } catch (err) {
        console.error('Error polling training status:', err);
        // Don't stop polling on error, just log it
      }
    }, 1000); // Poll every 1 second for real-time updates
  }

  async function startTraining() {
    if (!modelId) {
      error = 'Model ID is missing from URL parameters';
      return;
    }

    // Validate that model is fully configured
    if (!modelDetails?.layers_config?.length) {
      error = 'Model has no layers configured';
      return;
    }

    if (!modelDetails?.train_config) {
      error = 'Model has no training configuration';
      return;
    }

    if (!modelDetails?.dataset_config) {
      error = 'Model has no dataset configuration';
      return;
    }
    
    loading = true;
    training = true;
    error = null;
    stoppedByUser = false;

    try {
      const res = await client.mutate({
        mutation: TRAIN_MODEL,
        variables: { modelId },
        errorPolicy: 'all'
      });
      
      if (res.errors && res.errors.length > 0) {
        throw new Error(`GraphQL Error: ${res.errors.map(e => e.message).join(', ')}`);
      }
      
      if (!res.data?.train) {
        throw new Error('Failed to start training - no data returned');
      }
      
      // Initialize training status immediately to show that training has started
      trainingStatus = {
        epoch: 0,
        completed: false,
        loss: 0,
        accuracy: 0,
        started: true
      };
      
      // Start aggressive polling immediately
      startStatusPolling();
      
      // Do immediate status checks with exponential backoff
      const checkImmediately = async () => {
        for (let i = 0; i < 10; i++) {
          await new Promise(resolve => setTimeout(resolve, 200 * (i + 1))); // 200ms, 400ms, 600ms, etc.
          await checkTrainingStatus();
          
          // If we got real training data, break the loop
          if (trainingStatus && trainingStatus.epoch > 0) {
            break;
          }
        }
      };
      
      checkImmediately();
      
    } catch (err: any) {
      console.error('Training Error Details:', {
        message: err.message,
        networkError: err.networkError,
        graphQLErrors: err.graphQLErrors,
        fullError: err
      });
      
      if (err.networkError) {
        error = `Network Error: ${err.networkError.message}`;
      } else if (err.graphQLErrors && err.graphQLErrors.length > 0) {
        error = `GraphQL Error: ${err.graphQLErrors.map((e: any) => e.message).join(', ')}`;
      } else {
        error = err.message || err.toString() || 'Unknown error occurred';
      }
      training = false;
    } finally {
      loading = false;
    }
  }
</script>

<div class="start-training-container">
  <h1 class="heading">Start Training</h1>

  {#if !modelId}
    <div class="error-box">
      <p>No model ID provided in the URL.</p>
      <p class="mt-2">
        <a href="/create-model" class="checklist-links">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <div class="space-y-6">
      {#if modelDetails}
        <div class="model-overview">
          <h3 class="font-semibold model-info" style="margin-bottom: 12px; font-weight: 600;">Model Overview</h3>
          <div class="model-section">
            <div>
              <p class="model-info">
                <span>Model Name:</span> {modelDetails.name}
              </p>
              <p class="model-info">
                <span>Total Layers:</span> {modelDetails.layers_config?.length || 0}
              </p>
            </div>

            {#if modelDetails.train_config}
              <div>
                <p class="model-info">
                  <span>Epochs:</span> {modelDetails.train_config.epochs}
                </p>
                <p class="model-info">
                  <span>Optimizer:</span> {modelDetails.train_config.optimizer}
                </p>
                <p class="model-info">
                  <span>Learning Rate:</span> {modelDetails.train_config.optimizer_config?.lr}
                </p>
                <p class="model-info">
                  <span>Loss Function:</span> {modelDetails.train_config.loss_function}
                </p>
              </div>
            {/if}
          </div>

          {#if modelDetails.dataset_config}
            <div class="dataset-info">
              <p>
                <span>Dataset:</span> {modelDetails.dataset_config.name} 
                (Batch Size: {modelDetails.dataset_config.batch_size}, 
                Split: {(modelDetails.dataset_config.split_length?.[0] != null ? (modelDetails.dataset_config.split_length[0] * 100).toFixed(0) : 'N/A')}% train / 
                {(modelDetails.dataset_config.split_length?.[1] != null ? (modelDetails.dataset_config.split_length[1] * 100).toFixed(0) : 'N/A')}% test)
              </p>
            </div>
          {/if}
        </div>
      {/if}

      <!-- Configuration Status Check -->
      {#if modelDetails}
        <div class="checklist-section">
          <h3 class="checklist-title">Pre-Training Checklist</h3>
          <div>
            <div class="checklist-item">
              <div class="check-circle {modelDetails.layers_config?.length > 0 ? 'green' : 'red'}"></div>
              <span>Model Architecture: {modelDetails.layers_config?.length || 0} layers configured</span>
            </div>
            <div class="checklist-item">
              <div class="check-circle {modelDetails.train_config ? 'green' : 'red'}"></div>
              <span>Training Configuration: {modelDetails.train_config ? 'Ready' : 'Missing'}</span>
            </div>
            <div class="checklist-item">
              <div class="check-circle {modelDetails.dataset_config ? 'green' : 'red'}"></div>
              <span>Dataset Configuration: {modelDetails.dataset_config ? 'Ready' : 'Missing'}</span>
            </div>
          </div>
          
          {#if modelDetails.layers_config?.length > 0 && modelDetails.train_config && modelDetails.dataset_config}
            <div class="checklist-ready">
              ✅ All configurations complete. Ready to start training!
            </div>
          {:else}
            <div class="checklist-warning">
              ❌ Please complete all configurations before starting training.
              <div class="checklist-links" style="margin-top: 8px;">
                {#if !modelDetails.layers_config?.length}
                  <a href={`/create-model?modelId=${modelId}`}>Add Layers</a>
                {/if}
                {#if !modelDetails.train_config}
                  <a href={`/train-config?modelId=${modelId}`}>Configure Training</a>
                {/if}
                {#if !modelDetails.dataset_config}
                  <a href={`/dataset-config?modelId=${modelId}`}>Configure Dataset</a>
                {/if}
              </div>
            </div>
          {/if}
        </div>
      {/if}

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
              </div>
            </div>
          {:else}
            <div class="training-in-progress">
              <p>🔄 Training in Progress...</p>
              <div class="status-grid">
                <div>
                  <p>Current Epoch: {trainingStatus.epoch || 0} / {modelDetails?.train_config?.epochs || 'N/A'}</p>
                  <p>Status: Running</p>
                </div>
                <div>
                  <p>Current Loss: {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                  <p>Current Accuracy: {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                </div>
              </div>
            </div>
          {/if}
        </div>
      {/if}
      
      <!-- Training Controls -->
      <div class="training-controls">
        {#if !training && !trainingStatus?.completed}
          <button 
            on:click={startTraining}
            disabled={loading || !modelDetails?.layers_config?.length || !modelDetails?.train_config || !modelDetails?.dataset_config}
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
  {/if}
</div>