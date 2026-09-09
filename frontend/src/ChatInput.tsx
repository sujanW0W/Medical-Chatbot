import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"
import { LoaderCircle, Send } from "lucide-react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { fetchSessions, sendQuery } from "./queries"
import { useSession } from "./contexts/SessionContext"
import { cn } from "./lib/utils"

export default function ChatInput() {
    const { activeSessionId, setActiveSessionId } = useSession()

    const [query, setQuery] = useState("")

    const queryClient = useQueryClient()

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
        },
        onError: (error) => {
            console.log(error)
        }
    })

    const canSend = query.trim().length > 0 && !mutation.isPending

    const submit = () => {
        if (!canSend) return
        mutation.mutate({
            sessionId: activeSessionId,
            query: query
        })
    }

    return (
        <div className="flex flex-row gap-2 items-center m-auto">
            <Textarea
                placeholder="Ask Chatbot"
                className=" resize-none rounded-2xl bg-foreground/10 min-h-20 max-h-40 text-base focus:outline-none focus:ring-0 focus-visible:outline-none focus-visible:ring-0"
                value={query}
                disabled={mutation.isPending}
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
                    "p-2 rounded-full",
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