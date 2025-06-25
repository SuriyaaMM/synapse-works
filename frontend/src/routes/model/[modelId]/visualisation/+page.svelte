<script lang="ts">
  import { page } from '$app/stores';
  import { onMount, onDestroy } from 'svelte';
  import client from '$lib/apolloClient';
  import { START_TENSORBOARD } from '$lib/mutations';
  import { GET_MODEL, GET_TRAINING_STATUS } from '$lib/queries';

  import type { Model, TrainStatus } from '../../../../../../source/types';
  
  let modelId: string | null = null;
  let modelDetails: Model | null = null;
  let trainingStatus: TrainStatus | null = null;
  let tensorboardRunning = false;
  let tensorboardError: string | null = null;
  let loading = false;
  let tensorboardUrl = 'http://localhost:6006';
  let iframeLoaded = false;
  let connectionCheckInterval: NodeJS.Timeout | null = null;

  // Extract modelId from URL path
  $: {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    if (modelIndex !== -1 && modelIndex + 1 < pathParts.length) {
      modelId = pathParts[modelIndex + 1];
    } else {
      modelId = null;
    }
  }

  // Fetch model details and training status when modelId changes
  $: if (modelId) {
    fetchModelDetails();
    fetchTrainingStatus();
  }

  onMount(() => {
    if (modelId) {
      fetchModelDetails();
      fetchTrainingStatus();
    }
    checkTensorboardConnection();
    // Check connection every 30 seconds
    connectionCheckInterval = setInterval(checkTensorboardConnection, 2000);
  });

  onDestroy(() => {
    if (connectionCheckInterval) {
      clearInterval(connectionCheckInterval);
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
    }
  }

  async function fetchTrainingStatus() {
    if (!modelId) return;
    
    try {
      const response = await client.query({
        query: GET_TRAINING_STATUS,
        variables: { modelId },
        fetchPolicy: 'no-cache'
      });
      
      trainingStatus = response.data?.getTrainingStatus;
    } catch (err) {
      console.error('Error fetching training status:', err);
    }
  }

  function refreshTrainingStatus() {
    fetchTrainingStatus();
  }

  async function checkTensorboardConnection() {
    try {
      const response = await fetch(tensorboardUrl, { 
        method: 'HEAD',
        mode: 'no-cors'
      });
      tensorboardRunning = true;
      tensorboardError = null;
    } catch (err) {
      tensorboardRunning = false;
      tensorboardError = 'TensorBoard is not accessible at localhost:6006';
    }
  }

  async function startTensorboard() {
    if (!modelId) return;
    
    loading = true;
    tensorboardError = null;
    
    try {
      const response = await client.mutate({
        mutation: START_TENSORBOARD,
        variables: { modelId }
      });
      
      if (response.data?.startTensorboard) {
        tensorboardRunning = true;
        // Wait a moment for TensorBoard to start, then check connection
        setTimeout(checkTensorboardConnection, 3000);
      }
    } catch (err) {
      console.error('Error starting TensorBoard:', err);
      tensorboardError = 'Failed to start TensorBoard. Please check server logs.';
    } finally {
      loading = false;
    }
  }

  function handleIframeLoad() {
    iframeLoaded = true;
  }

  function handleIframeError() {
    tensorboardRunning = false;
    tensorboardError = 'Failed to load TensorBoard. Please ensure it is running.';
  }
</script>

<div class="container mx-auto p-6">
  <h1 class="text-3xl font-bold mb-6">Model Visualization</h1>

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
      <!-- Training Status Card -->
      {#if trainingStatus}
        <div class="bg-gray-50 p-6 rounded-lg border border-gray-200">
          <div class="flex justify-between items-center mb-4">
            <h2 class="text-xl font-semibold text-gray-800">🏃‍♂️ Training Status</h2>
            <button 
              on:click={refreshTrainingStatus}
              class="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            >
              🔄 Refresh
            </button>
          </div>
          
          {#if trainingStatus.completed}
            <div class="bg-green-100 p-4 rounded-lg border border-green-300">
              <p class="font-semibold text-green-800 mb-2">✅ Training Completed Successfully!</p>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div class="text-green-700">
                  <p><span class="font-semibold">Total Epochs:</span> {trainingStatus.epoch}</p>
                </div>
                <div class="text-green-700">
                  <p><span class="font-semibold">Final Loss:</span> {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                </div>
                <div class="text-green-700">
                  <p><span class="font-semibold">Final Accuracy:</span> {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                </div>
              </div>
            </div>
          {:else}
            <div class="bg-yellow-100 p-4 rounded-lg border border-yellow-300">
              <p class="font-semibold text-yellow-800 mb-2">⏳ Training Status</p>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div class="text-yellow-700">
                  <p><span class="font-semibold">Current Epoch:</span> {trainingStatus.epoch || 0}</p>
                </div>
                <div class="text-yellow-700">
                  <p><span class="font-semibold">Current Loss:</span> {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                </div>
                <div class="text-yellow-700">
                  <p><span class="font-semibold">Current Accuracy:</span> {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                </div>
              </div>
              <p class="text-yellow-600 text-sm mt-2">
                Training is {trainingStatus.started ? 'in progress' : 'not started yet'}
              </p>
            </div>
          {/if}
        </div>
      {:else}
        <div class="bg-gray-100 p-6 rounded-lg border border-gray-300">
          <h2 class="text-xl font-semibold text-gray-600 mb-2">📈 Training Status</h2>
          <p class="text-gray-600">No training data available yet.</p>
          <p class="text-sm text-gray-500 mt-1">
            Start training your model to see visualization data here.
          </p>
        </div>
      {/if}

      <!-- TensorBoard Visualization Section -->
      <div class="bg-purple-50 p-6 rounded-lg border border-purple-200">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xl font-semibold text-purple-800">📊 TensorBoard Visualization</h2>
          <div class="flex gap-2">
            <div class="flex items-center gap-2">
              <div class="w-3 h-3 rounded-full {tensorboardRunning ? 'bg-green-500' : 'bg-red-500'}"></div>
              <span class="text-sm {tensorboardRunning ? 'text-green-700' : 'text-red-700'}">
                {tensorboardRunning ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button 
              on:click={startTensorboard}
              disabled={loading || tensorboardRunning}
              class="px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? '⏳ Starting...' : '🚀 Start TensorBoard'}
            </button>
            <button 
              on:click={checkTensorboardConnection}
              class="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            >
              🔄 Check Connection
            </button>
          </div>
        </div>
        
        <div class="space-y-4">
          {#if tensorboardRunning}
            <div class="border border-gray-300 rounded-lg overflow-hidden bg-white">
              <iframe 
                src={tensorboardUrl}
                class="w-full h-96 md:h-[600px] lg:h-[700px]"
                title="TensorBoard Visualization"
                frameborder="0"
                on:load={handleIframeLoad}
                on:error={handleIframeError}
              ></iframe>
            </div>
          {:else}
            <div class="border border-gray-300 rounded-lg p-8 bg-gray-100 text-center">
              <div class="text-gray-500 mb-4">
                <svg class="w-16 h-16 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                </svg>
                <p class="text-lg font-medium">TensorBoard Not Available</p>
                <p class="text-sm">Click "Start TensorBoard" to launch the visualization server</p>
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>