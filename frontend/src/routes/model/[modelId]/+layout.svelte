<script lang="ts">
  import type { LayoutData } from './$types';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { derived } from 'svelte/store';

  export let data: LayoutData;
  console.log(data);
  $: modelId = data.modelId;
  $: modelName = data.modelName;

  // Reactive tabs update when modelId changes
  $: tabs = [
    { id: 'dataset-config', label: 'Configure Dataset', path: `/model/${modelId}/dataset-config` },
    { id: 'graph-construction', label: 'Graph Construction', path: `/model/${modelId}/graph-construction` },
    { id: 'training-config', label: 'Configure Training', path: `/model/${modelId}/training-config` },
    { id: 'train-model', label: 'Train the Model', path: `/model/${modelId}/train-model` },
    { id: 'visualisation', label: 'Visualisation', path: `/model/${modelId}/visualisation` },
    { id: 'save-model', label: 'Save Model', path: `/model/${modelId}/save-model` }
  ];

  // Determine current step based on URL path
  const currentStep = derived(page, ($page) => {
    if ($page.url.pathname.includes('/layer-config')) return 'layer-config';
    if ($page.url.pathname.includes('/graph-construction')) return 'graph-construction';
    if ($page.url.pathname.includes('/dataset-config')) return 'dataset-config';
    if ($page.url.pathname.includes('/training-config')) return 'training-config';
    if ($page.url.pathname.includes('/train-model')) return 'train-model';
    if ($page.url.pathname.includes('/save-model')) return 'save-model';
    if ($page.url.pathname.includes('/visualisation')) return 'visualisation';
    return '';
  });
</script>

<div class="h-screen flex flex-col p-6">
  <div class="flex justify-between items-center mb-1">
    <h1 class="text-2xl font-bold">{modelName}</h1>
    <button on:click={() => goto('/create-model')} class="border px-4 py-2 rounded hover:bg-gray-100">
      Create Another Model
    </button>
  </div>

  <!-- Horizontal Tabs -->
  <div class="border-b border-gray-200 mb-1">
    <nav class="flex space-x-8">
      {#each tabs as tab}
        <button
          class="py-2 px-1 border-b-2 font-medium text-sm transition-colors duration-200
                 {($currentStep === tab.id)
                   ? 'border-blue-500 text-blue-600'
                   : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}"
          on:click={() => goto(tab.path)}
        >
          {tab.label}
        </button>
      {/each}
    </nav>
  </div>

  <!-- Main Content -->
  <div class="flex-1 border rounded-lg shadow overflow-hidden">
    <div class="h-full p-8 overflow-y-auto">
      <slot />
    </div>
  </div>
</div>
