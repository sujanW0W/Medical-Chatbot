import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"
import { LoaderCircle, Send } from "lucide-react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { fetchLatestJob, fetchSessions, sendQuery } from "./queries"
import { useSession } from "./contexts/SessionContext"
import { cn } from "./lib/utils"

export default function ChatInput() {
    const { activeSessionId, setActiveSessionId } = useSession()

    const [query, setQuery] = useState("")

    const queryClient = useQueryClient()

    // Same query key ChatSection polls, sharing its cache - but no
    // refetchInterval here. ChatSection stays the sole poller; this 
    // just reads the same cache entry and re-renders whenever that 
    // polling updates it.
    const { data: latestJob } = useQuery({
        queryKey: ["latest-job", activeSessionId],
        queryFn: () => fetchLatestJob(activeSessionId),
        enabled: !!activeSessionId,
    })
    const isJobActive = latestJob?.status === "queued" || latestJob?.status === "processing"

    const mutation = useMutation({
        mutationFn: sendQuery,
        onSuccess: async () => {
            setQuery("")

            let sessionId = activeSessionId

            if (!sessionId) {
                const sessions = await queryClient.fetchQuery({ queryKey: ["sessions"], queryFn: fetchSessions })
                sessionId = sessions?.[0]?.id
                setActiveSessionId(sessionId)
            }

            queryClient.invalidateQueries({ queryKey: ["messages", sessionId] })
            queryClient.invalidateQueries({ queryKey: ["latest-job", sessionId] })
        },
        onError: (error) => {
            toast.error(error.message || "Something went wrong. Please try again.")
        },
    })

    const canSend = query.trim().length > 0 && !mutation.isPending && !isJobActive

    const submit = () => {
        if (!canSend) return
        mutation.mutate({
            sessionId: activeSessionId,
            query: query
        })
    }

    return (
        <div className="flex flex-row gap-2 items-center m-auto w-full">
            <Textarea
                placeholder={isJobActive ? "Waiting for a response..." : "Ask Chatbot"}
                className=" resize-none rounded-2xl bg-foreground/10 min-h-20 max-h-40 text-base focus:outline-none focus:ring-0 focus-visible:outline-none focus-visible:ring-0"
                value={query}
                disabled={mutation.isPending || isJobActive}
                onChange={event => setQuery(event.target.value)}
                onKeyDown={event => {
                    if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault()
                        submit()
                    }
                }}
            />
            <div
                className={cn(
                    "p-2 rounded-full transition-colors",
                    canSend ? "cursor-pointer hover:bg-foreground/15" : "cursor-not-allowed opacity-50"
                )}
                onClick={submit}
            >
                {mutation.isPending
                    ? <LoaderCircle size={20} className="animate-spin" />
                    : <Send size={20} />
                }
            </div>
        </div>
    )
}