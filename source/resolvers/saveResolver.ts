import { Model } from "../types/modelTypes.js";
import { setModel } from "../resolvers.js";
import { enqueueMessage } from "../redisClient.js";
import { writeFile, readFile, mkdir } from "fs/promises";
import path  from "path";

export async function serialize(model: Model, dirname: string = "./savefile") {

    if(!model) throw new Error(`[synapse]: Model was empty! nothing to serialize`);

    const jsonified_model = JSON.stringify(model, null, 4);
    const file_path = path.join(dirname, `${model.id}.json`);

    await mkdir(dirname, { recursive: true });

    await writeFile(file_path, jsonified_model, 'utf-8');
    console.log(`[synapse]: Serialized Model(id = ${model.id} to ${file_path}`);

    return true;
}

export async function deserialize(model_id: string, dirname: string = "./savefile"){

    const file_path = path.join(dirname, `${model_id}.json`);

    const models_buffer = await readFile(file_path, 'utf-8');
    const model = JSON.parse(models_buffer);

    console.log(`[synapse]: De-Serialized from ${file_path}`);
    setModel(model);
    console.log(`[synapse]: Updated Model: ${JSON.stringify(model)}`)
    return model;
}

export async function saveResolver(model: Model, dirname: string = "./savefile"){
    // push message to redis
    console.log(`[synapse][graphql]: Appending SERIALIZE_MODEL Event to redis Queue`)
    const message = {
        event_type: "SERIALIZE_MODEL",
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    return serialize(model);
}

export async function loadModelResolver(model_id: string){
    // push message to redis
    console.log(`[synapse][graphql]: Appending DESERIALIZE_MODEL to redis Queue`)
    const message = {
        event_type: "DESERIALIZE_MODEL",
        model_id: model_id,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    return await deserialize(model_id);
}