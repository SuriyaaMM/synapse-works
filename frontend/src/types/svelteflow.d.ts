declare module '@xyflow/svelte' {
  import type { SvelteComponentTyped } from 'svelte';

  export interface SvelteFlowProps {
    nodes?: any[];
    edges?: any[];
    nodeTypes?: any;
    edgeTypes?: any;
    fitView?: boolean;
    // Event handler props (not events)
    onnodeclick?: (event: { node: any, event: MouseEvent }) => void;
    onedgeclick?: (event: { edge: any, event: MouseEvent }) => void;
    onconnect?: (connection: { source: string; target: string }) => void;
    onnodeschange?: (event: CustomEvent<any[]>) => void;
    onselectionchange?: (params: { nodes: any[], edges: any[] }) => void;
    onpaneclick?: (event: { event: MouseEvent }) => void;
    // Add other common event handlers as needed
    onnodedragstart?: (event: { node: any, event: MouseEvent }) => void;
    onnodedrag?: (event: { node: any, event: MouseEvent }) => void;
    onnodedragstop?: (event: { node: any, event: MouseEvent }) => void;
  }

  export class SvelteFlow extends SvelteComponentTyped<
    SvelteFlowProps,
    {}, // No custom events since they're handled via props
    any
  > {}
}