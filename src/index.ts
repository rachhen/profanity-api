import { Hono } from "hono";
import { cors } from "hono/cors";
import { env } from "hono/adapter";
import { Index } from "@upstash/vector";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

type Env = {
  VECTOR_URL: string;
  VECTOR_TOKEN: string;
};

const app = new Hono();

const WHITELIST = ["swear"];
const PROFANITY_THRESHOLD = 0.86;
const semanticSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 25,
  separators: [" "],
  chunkOverlap: 12,
});

app.use(cors());

app.post("/", async (c) => {
  if (c.req.header("Content-Type") !== "application/json") {
    return c.json({ error: "JSON body expected" }, 406);
  }

  try {
    const { VECTOR_URL, VECTOR_TOKEN } = env<Env>(c);

    const index = new Index({
      url: VECTOR_URL,
      token: VECTOR_TOKEN,
      cache: false,
    });

    let { message } = await c.req.json<{ message?: string }>();

    if (!message) {
      return c.json({ error: "Message argument is required." }, 400);
    }

    if (message.length > 1000) {
      return c.json(
        { error: "Message can only be at most 1000 characters" },
        413
      );
    }

    message = message
      .split(/\s/)
      .filter((word) => !WHITELIST.includes(word.toLowerCase()))
      .join(" ");

    const [semanticChunks, wordChunks] = await Promise.all([
      splitTextIntoSemantics(message),
      splitTextIntoWords(message),
    ]);

    const flaggedFor = new Set<{ score: number; text: string }>();

    const vectorRes = await Promise.all([
      ...wordChunks.map(async (wordChunk) => {
        const [vector] = await index.query({
          topK: 1,
          data: wordChunk,
          includeMetadata: true,
        });

        if (vector && vector.score > 0.95) {
          flaggedFor.add({
            score: vector.score,
            text: vector.metadata!.text as string,
          });
        }

        return { score: 0 };
      }),
      ...semanticChunks.map(async (semanticChunk) => {
        const [vector] = await index.query({
          topK: 1,
          data: semanticChunk,
          includeMetadata: true,
        });

        if (vector && vector.score > PROFANITY_THRESHOLD) {
          flaggedFor.add({
            score: vector.score,
            text: vector.metadata!.text as string,
          });
        }

        return vector;
      }),
    ]);

    if (flaggedFor.size > 0) {
      const sorted = Array.from(flaggedFor).sort((a, b) =>
        a.score > b.score ? -1 : 1
      )[0];

      return c.json({
        isProfanity: true,
        score: sorted.score,
        flaggedFor: sorted.text,
      });
    } else {
      const mostProfanityChunk = vectorRes.sort((a, b) =>
        a.score > b.score ? -1 : 1
      )[0];

      return c.json({
        isProfanity: false,
        score: mostProfanityChunk.score,
      });
    }
  } catch (err) {
    console.error(err);

    return c.json({ error: "Something went wrong" }, 500);
  }
});

function splitTextIntoWords(text: string) {
  return text.split(/\s/);
}

async function splitTextIntoSemantics(text: string) {
  if (text.split(/\s/).length === 1) return [];

  const documents = await semanticSplitter.createDocuments([text]);
  const chunks = documents.map((chunk) => chunk.pageContent);
  return chunks;
}

export default app;
