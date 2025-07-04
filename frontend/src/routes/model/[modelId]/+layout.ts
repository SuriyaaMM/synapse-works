import { fetchModelDetails, modelDetails } from "./modelDetails";
import type { LayoutLoad } from './$types';

export const load: LayoutLoad = async ({ params }) => {
    const { modelId } = params;

    await fetchModelDetails();

    let name: string | null = null;

    modelDetails.subscribe((model: any) => {
        name = model?.name || 'Unnamed Model';
    })();

    return {
        modelId,
        modelName: name
    };
};
