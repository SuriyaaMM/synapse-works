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
      <!-- Model Overview Card -->
      

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
        <h2 class="text-xl font-semibold text-purple-800 mb-4">📊 TensorBoard Visualization</h2>
        
        <div class="space-y-4">
          <div class="border border-gray-300 rounded-lg overflow-hidden">
            <iframe 
              src="http://localhost:6006" 
              class="w-full h-96 md:h-[600px] lg:h-[700px]"
              title="TensorBoard Visualization"
              frameborder="0"
            ></iframe>
          </div>

          {#if tensorboardError}
            <div class="bg-red-100 border border-red-400 text-red-700 rounded-lg p-4">
              <p class="font-semibold">❌ TensorBoard Error:</p>
              <p class="text-sm mt-1">{tensorboardError}</p>
              <div class="mt-3 p-2 bg-red-50 rounded text-xs">
                <p class="font-semibold">Troubleshooting:</p>
                <ul class="list-disc list-inside mt-1 space-y-1">
                  <li>Make sure TensorBoard is installed: <code class="bg-red-200 px-1 rounded">pip install tensorboard</code></li>
                  <li>Check if port 6006 is available</li>
                  <li>Verify that the tbsummary/ directory exists</li>
                  <li>Try restarting TensorBoard if it's already running</li>
                </ul>
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>