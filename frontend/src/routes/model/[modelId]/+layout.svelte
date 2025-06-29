<script lang="ts">
  import type { LayoutData } from './$types';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { derived } from 'svelte/store';

  export let data: LayoutData;
  $: modelId = data.modelId;

  const currentStep = derived(page, ($page) => {
    if ($page.url.pathname.includes('/layer-config')) return 'layer-config';
    if ($page.url.pathname.includes('/dataset-config')) return 'dataset-config';
    if ($page.url.pathname.includes('/training-config')) return 'training-config';
    if ($page.url.pathname.includes('/train-model')) return 'train-model';
    if ($page.url.pathname.includes('/save-model')) return 'save-model';
    if ($page.url.pathname.includes('/visualisation')) return 'visualisation';
    return '';
  });
</script>

<div class="h-screen flex flex-col p-6">
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-2xl font-bold">Model Configuration</h1>
    <button on:click={() => goto('/create-model')} class="border px-4 py-2 rounded hover:bg-gray-100">Create Another Model</button>
  </div>

  <div class="flex flex-1 border rounded-lg shadow overflow-hidden">
    <!-- Sidebar Stepper -->
    <div class="w-64 bg-gray-50 border-r p-6 space-y-6">
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'layer-config') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/layer-config`)}
      >
         Configure Layers
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'training-config') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/training-config`)}
      >
         Configure Training
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'dataset-config') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/dataset-config`)}
      >
         Configure Dataset
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'train-model') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/train-model`)}
      >
         Train the Model
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'save-model') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/save-model`)}
      >
         Save Model
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'visualisation') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/visualisation`)}
      >
         Visualisation
      </button>
    </div>

    <!-- Main Content -->
    <div class="flex-1 p-8 overflow-y-auto">
      <slot />
    </div>
  </div>
</div>
