import { ApolloServer } from "@apollo/server";
import { startStandaloneServer } from "@apollo/server/standalone";
import { typeDefs } from "./schema.js";
import { resolvers } from "./resolvers.js";
import { connectRedis } from "./redisClient.js";
import { tensorboardProcess } from "./resolvers.js";

connectRedis();

const server = new ApolloServer({
    typeDefs,
    resolvers
})

const {url} = await startStandaloneServer(
    server, {
        listen: {port : 4000}
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