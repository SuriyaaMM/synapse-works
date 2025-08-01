import { ApolloServer } from "@apollo/server";
import { startStandaloneServer } from "@apollo/server/standalone";
import { schema } from "./schema.js";
import { connectRedis, PORT } from "./redisClient.js";
import { tensorboardProcess } from "./resolvers.js";

await connectRedis();

const server = new ApolloServer({ schema })

const {url} = await startStandaloneServer(
    server, {
        listen: { port: PORT}

    }
)

console.log("Started server http://localhost:4000")

const cleanup = () => {
    if (tensorboardProcess) {
        console.log('[Cleanup]: Killing TensorBoard before exit.');
        tensorboardProcess.kill('SIGTERM');
    }
};

process.on('exit', cleanup);
process.on('SIGINT', () => {
    cleanup();
    process.exit(0);
});
process.on('SIGTERM', () => {
    cleanup();
    process.exit(0);
});
process.on('uncaughtException', (err) => {
    console.error('[Uncaught Exception]', err);
    cleanup();
    process.exit(1);
});