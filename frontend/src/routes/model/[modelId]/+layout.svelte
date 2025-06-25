<script lang="ts">
  import type { LayoutData } from './$types';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { derived } from 'svelte/store';

  export let data: LayoutData;
  $: modelId = data.modelId;

  const currentStep = derived(page, ($page) => {
    if ($page.url.pathname.includes('/layers')) return 'layers';
    if ($page.url.pathname.includes('/dataset')) return 'dataset';
    if ($page.url.pathname.includes('/training')) return 'training';
    if ($page.url.pathname.includes('/train')) return 'train';
    if ($page.url.pathname.includes('/save-load')) return 'save-load';
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
                {($currentStep === 'layers') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/layers`)}
      >
         Configure Layers
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'training') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/training`)}
      >
         Configure Training
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'dataset') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/dataset`)}
      >
         Configure Dataset
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'train') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/train`)}
      >
         Train the Model
      </button>
      <button
        class="w-full text-left py-2 px-4 rounded-lg transition 
                hover:bg-gray-200 
                {($currentStep === 'save-load') ? 'bg-blue-100 font-semibold' : ''}"
        on:click={() => goto(`/model/${modelId}/save-load`)}
      >
         Save and Load Model
      </button>
    </div>

    <!-- Main Content -->
    <div class="flex-1 p-8 overflow-y-auto">
      <slot />
    </div>
  </div>
</div>
