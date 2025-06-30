<script lang="ts">
  import { page } from '$app/stores';
  import { onMount, onDestroy } from 'svelte';
  import client from '$lib/apolloClient';
  import { START_TENSORBOARD } from '$lib/mutations';
  import { GET_MODEL, GET_TRAINING_STATUS } from '$lib/queries';
  
  import './visualization.css';

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
    // Check connection every 2 seconds
    connectionCheckInterval = setInterval(checkTensorboardConnection, 500);
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
        setTimeout(checkTensorboardConnection, 2000);
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

<div class="container">
  <h1 class="title">Model Visualization</h1>

  {#if !modelId}
    <div class="alert">
      <p>No model ID provided in the URL.</p>
      <p>
        <a href="/create-model" class="link">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <div class="space-y-6">
      {#if trainingStatus}
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">🏃‍♂️ Training Status</h2>
            <button on:click={refreshTrainingStatus} class="button">
              🔄 Refresh
            </button>
          </div>

          {#if trainingStatus.completed}
            <div class="success-card">
              <p class="success-title">✅ Training Completed Successfully!</p>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div class="success-text">
                  <p><span class="font-semibold">Total Epochs:</span> {trainingStatus.epoch}</p>
                </div>
                <div class="success-text">
                  <p><span class="font-semibold">Final Loss:</span> {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                </div>
                <div class="success-text">
                  <p><span class="font-semibold">Final Accuracy:</span> {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                </div>
              </div>
            </div>
          {:else}
            <div class="warning-card">
              <p class="warning-title">⏳ Training Status</p>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div class="warning-text">
                  <p><span class="font-semibold">Current Epoch:</span> {trainingStatus.epoch || 0}</p>
                </div>
                <div class="warning-text">
                  <p><span class="font-semibold">Current Loss:</span> {trainingStatus.loss?.toFixed(4) || 'N/A'}</p>
                </div>
                <div class="warning-text">
                  <p><span class="font-semibold">Current Accuracy:</span> {trainingStatus.accuracy ? (trainingStatus.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                </div>
              </div>
              <p class="warning-text text-sm mt-2">
                Training is {trainingStatus.started ? 'in progress' : 'not started yet'}
              </p>
            </div>
          {/if}
        </div>
      {:else}
        <div class="gray-card">
          <h2 class="gray-title">📈 Training Status</h2>
          <p class="gray-text">No training data available yet.</p>
          <p class="gray-subtext">
            Start training your model to see visualization data here.
          </p>
        </div>
      {/if}

      <div class="purple-card">
        <div class="card-header">
          <h2 class="purple-title">📊 TensorBoard Visualization</h2>
          <div class="flex gap-2">
            <div class="status">
              <div class="status-dot {tensorboardRunning ? 'connected' : 'disconnected'}"></div>
              <span class="{tensorboardRunning ? 'connected' : 'disconnected'} text-sm">
                {tensorboardRunning ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button
              on:click={startTensorboard}
              disabled={loading || tensorboardRunning}
              class="button {loading || tensorboardRunning ? 'disabled' : ''}"
            >
              {loading ? '⏳ Starting...' : '🚀 Start TensorBoard'}
            </button>
            <button on:click={checkTensorboardConnection} class="button">
              🔄 Check Connection
            </button>
          </div>
        </div>

        <div class="space-y-4">
          {#if tensorboardRunning}
            <div class="iframe-container">
              <iframe
                src={tensorboardUrl}
                class="iframe"
                title="TensorBoard Visualization"
                frameborder="0"
                on:load={handleIframeLoad}
                on:error={handleIframeError}
              ></iframe>
            </div>
          {:else}
            <div class="placeholder">
              <svg class="svg-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z">
                </path>
              </svg>
              <p class="text-lg font-medium">TensorBoard Not Available</p>
              <p class="text-sm">Click "Start TensorBoard" to launch the visualization server</p>
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>