import type { RequestType, ResponseType } from "@/types";

export async function request<T = unknown>({
    endpoint,
    body,
    method = "GET",
    headers = {},
}: RequestType): Promise<ResponseType<T>> {
    try {
        const URL = `${import.meta.env.VITE_BASE_URL}/${endpoint}`;

        const response = await fetch(URL, {
            method: method,
            headers: {
                "Content-Type": "application/json",
                ...headers,
            },
            body: method !== "GET" && body ? JSON.stringify(body) : undefined,
        });

        // const contentType = response.headers.get("content-type");

        // const res = contentType?.includes("application/json")
        //     ? await response.json()
        //     : await response.text();

        const res = await response.json();

        if (!response.ok) {
            return {
                status: response.status,
                error: res.error || "Request Failed",
            };
        }

        return {
            status: response.status,
            data: res.data as T,
        };
    } catch (error) {
        return {
            status: 500,
            error: error instanceof Error ? error.message : "Unknown error",
        };
    }
}
