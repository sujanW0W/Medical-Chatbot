import { request } from "./api/requests";
import type { ChatResponse, Message, Session } from "./types";

export const fetchSessions = async () => {
    const res = await request<Array<Session>>({ endpoint: "sessions/" });
    let data = res.data;
    data =
        data?.sort(
            (a, b) => -(Date.parse(a.updated_at) - Date.parse(b.updated_at)),
        ) || [];
    return data;
};

export const fetchMessages = async (sessionId: string | undefined) => {
    if (!sessionId) return [];

    const res = await request<Array<Message>>({
        endpoint: `sessions/${sessionId}/conversations`,
    });
    const data = res.data;

    return data || [];
};

export const sendQuery = async ({
    sessionId,
    query,
}: {
    sessionId: string | undefined;
    query: string;
}) => {
    if (!query) {
        return;
    }

    const res = await request<ChatResponse>({
        endpoint: sessionId ? `chat/ask/${sessionId}` : "chat/ask",
        method: "POST",
        body: {
            content: query,
        },
    });

    const data = res.data;

    return data;
};
