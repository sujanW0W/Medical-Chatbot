export interface Session {
    id: string;
    name: string;
}

export interface Message {
    id: string;
    role: string;
    content: string;
    timestamp: string;
}

export interface RequestType {
    endpoint: string;
    method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
    headers?: HeadersInit;
    body?: Record<string, string>;
}

export interface ResponseType<T> {
    status: number;
    data?: T;
    error?: string;
}

export interface Session {
    id: string;
    name: string;
    created_at: string;
    updated_at: string;
}

export interface ChatResponse {
    job_id: string;
}

export type JobStatus = "queued" | "processing" | "failed" | "success" | "cancelled";

export interface Job {
    id: string;
    conversation_id: string;
    status: JobStatus;
    result_conversation_id: string | null;
    error_conversation_id: string | null;
    created_at: string;
    updated_at: string;
}
