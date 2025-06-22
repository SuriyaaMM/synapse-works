<script lang="ts">
  import { page } from '$app/stores';
  import { onMount, onDestroy } from 'svelte';
  import client from '$lib/apolloClient';
  import { TRAIN_MODEL } from '$lib/mutations';
  import { GET_MODEL, GET_TRAINING_STATUS } from '$lib/queries';

  import type { Model, TrainStatus } from '../../../../../../source/types';
  
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

  function startSlowStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    
    statusInterval = setInterval(async () => {
      try {
        const response = await client.query({
          query: GET_TRAINING_STATUS,
          variables: { modelId },
          fetchPolicy: 'no-cache'
        });
        
        const newStatus = response.data?.getTrainingStatus;
        
        if (newStatus) {
          trainingStatus = newStatus;
        }
        
        // Stop polling if training is completed
        if (trainingStatus?.completed) {
          clearInterval(statusInterval);
          statusInterval = null;
          training = false;
        }
      } catch (err) {
        console.error('Error polling training status:', err);
      }
    }, 2000);
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
        status: 'Training started - waiting for first epoch...',
        completed: false,
        loss: null,
        accuracy: null
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
        error = `GraphQL Error: ${err.graphQLErrors.map(e => e.message).join(', ')}`;
      } else {
        error = err.message || err.toString() || 'Unknown error occurred';
      }
      training = false;
    } finally {
      loading = false;
    }
  }
</script>

<div class="container mx-auto p-6">
  <h1 class="text-3xl font-bold mb-6">Start Training</h1>

  {#if !modelId}
    <div class="p-4 bg-red-100 border border-red-400 text-red-700 rounded">
      <p>No model ID provided in the URL.</p>
      <p class="mt-2">
        <a href="/create-model" class="text-blue-600 underline">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <div class="space-y-6">
      {#if modelDetails}
        <div class="bg-blue-50 p-4 rounded-md">
          <h3 class="font-semibold text-blue-800 mb-2">Model Overview</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <p class="text-blue-700">
                <span class="font-semibold">Model Name:</span> {modelDetails.name}
              </p>
              <p class="text-blue-700">
                <span class="font-semibold">Total Layers:</span> {modelDetails.layers_config?.length || 0}
              </p>
            </div>
            
            {#if modelDetails.train_config}
              <div>
                <p class="text-blue-700">
                  <span class="font-semibold">Epochs:</span> {modelDetails.train_config.epochs}
                </p>
                <p class="text-blue-700">
                  <span class="font-semibold">Optimizer:</span> {modelDetails.train_config.optimizer}
                </p>
                <p class="text-blue-700">
                  <span class="font-semibold">Learning Rate:</span> {modelDetails.train_config.optimizer_config?.lr}
                </p>
                <p class="text-blue-700">
                  <span class="font-semibold">Loss Function:</span> {modelDetails.train_config.loss_function}
                </p>
              </div>
            {/if}
          </div>

          {#if modelDetails.dataset_config}
            <div class="mt-3 pt-3 border-t border-blue-200">
              <p class="text-blue-700 text-sm">
                <span class="font-semibold">Dataset:</span> {modelDetails.dataset_config.name} 
                (Batch Size: {modelDetails.dataset_config.batch_size}, 
                Split: {(modelDetails.dataset_config.split_length?.[0] * 100).toFixed(0)}% train / 
                {(modelDetails.dataset_config.split_length?.[1] * 100).toFixed(0)}% test)
              </p>
            </div>
          {/if}
        </div>
      {/if}

      <!-- Configuration Status Check -->
      {#if modelDetails}
        <div class="bg-gray-50 p-4 rounded-md">
          <h3 class="font-semibold text-gray-800 mb-3">Pre-Training Checklist</h3>
          <div class="space-y-2">
            <div class="flex items-center">
              <div class="w-4 h-4 rounded-full {modelDetails.layers_config?.length > 0 ? 'bg-green-500' : 'bg-red-500'} mr-2"></div>
              <span class="text-sm">Model Architecture: {modelDetails.layers_config?.length || 0} layers configured</span>
            </div>
            <div class="flex items-center">
              <div class="w-4 h-4 rounded-full {modelDetails.train_config ? 'bg-green-500' : 'bg-red-500'} mr-2"></div>
              <span class="text-sm">Training Configuration: {modelDetails.train_config ? 'Ready' : 'Missing'}</span>
            </div>
            <div class="flex items-center">
              <div class="w-4 h-4 rounded-full {modelDetails.dataset_config ? 'bg-green-500' : 'bg-red-500'} mr-2"></div>
              <span class="text-sm">Dataset Configuration: {modelDetails.dataset_config ? 'Ready' : 'Missing'}</span>
            </div>
          </div>
          
          {#if modelDetails.layers_config?.length > 0 && modelDetails.train_config && modelDetails.dataset_config}
            <div class="mt-3 p-2 bg-green-100 rounded border-l-4 border-green-500">
              <p class="text-sm text-green-700 font-medium">
                ✅ All configurations complete. Ready to start training!
              </p>
            </div>
          {:else}
            <div class="mt-3 p-2 bg-red-100 rounded border-l-4 border-red-500">
              <p class="text-sm text-red-700 font-medium">
                ❌ Please complete all configurations before starting training.
              </p>
              <div class="mt-2 space-x-2">
                {#if !modelDetails.layers_config?.length}
                  <a href={`/create-model?modelId=${modelId}`} class="text-xs bg-blue-600 text-white px-2 py-1 rounded">
                    Add Layers
                  </a>
                {/if}
                {#if !modelDetails.train_config}
                  <a href={`/train-config?modelId=${modelId}`} class="text-xs bg-blue-600 text-white px-2 py-1 rounded">
                    Configure Training
                  </a>
                {/if}
                {#if !modelDetails.dataset_config}
                  <a href={`/dataset-config?modelId=${modelId}`} class="text-xs bg-blue-600 text-white px-2 py-1 rounded">
                    Configure Dataset
                  </a>
                {/if}
              </div>
            </div>
          {/if}
        </div>
      {/if}

      <!-- Training Status -->
      {#if trainingStatus}
        <div class="bg-yellow-50 p-4 rounded-md border-l-4 border-yellow-500">
          <h3 class="font-semibold text-yellow-800 mb-3">Training Status</h3>
          
          {#if trainingStatus.completed}
            <div class="text-green-700">
              <p class="font-medium">✅ Training Completed!</p>
              <div class="mt-2 text-sm">
                <p>Final Epoch: {trainingStatus.epoch}</p>
                <p>Final Loss: {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                <p>Final Accuracy: {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
              </div>
            </div>
          {:else}
            <div class="text-yellow-700">
              <p class="font-medium">🔄 Training in Progress...</p>
              <div class="mt-3 text-sm grid grid-cols-2 gap-4">
                <div>
                  <p>Current Epoch: {trainingStatus.epoch || 0} / {modelDetails?.train_config?.epochs || 'N/A'}</p>
                  <p>Status: {trainingStatus.status || 'Running'}</p>
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
      <div class="flex space-x-3 pt-6">
        {#if !training && !trainingStatus?.completed}
          <button 
            on:click={startTraining}
            disabled={loading || !modelDetails?.layers_config?.length || !modelDetails?.train_config || !modelDetails?.dataset_config}
            class="px-8 py-4 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-lg hover:from-green-700 hover:to-green-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            {loading ? 'Starting Training...' : '🚀 Start Training'}
          </button>
        {:else if training && !trainingStatus?.completed}
          <button 
            on:click={stopTraining}
            class="px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg hover:from-red-700 hover:to-red-800 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            🛑 Stop Training
          </button>
          
          {#if stoppedByUser}
            <div class="mt-3 p-3 bg-orange-100 border border-orange-400 text-orange-700 rounded">
              <p class="text-sm">⚠️ Training stopped by user</p>
            </div>
          {/if}
        {/if}
      </div>

      {#if error}
        <div class="p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          <h4 class="font-semibold">Training Error:</h4>
          <p class="mt-1">{error}</p>
        </div>
      {/if}
    </div>
  {/if}
</div>