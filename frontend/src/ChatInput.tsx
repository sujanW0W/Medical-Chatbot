import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"
import { Send } from "lucide-react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { sendQuery } from "./queries"
import { useSession } from "./contexts/SessionContext"

export default function ChatInput() {
    const {activeSessionId, setActiveSessionId} = useSession()

    const [query, setQuery] = useState("")

    const queryClient = useQueryClient()

    const mutation = useMutation({
        mutationFn: sendQuery,
        onSuccess: (data) => {
            setQuery("")
            setActiveSessionId(data?.session_id)
            if(!activeSessionId)
                queryClient.invalidateQueries({queryKey: ["sessions"]})
            queryClient.invalidateQueries({queryKey: ["messages", activeSessionId]})
        }
    })

    return (
        <div className="flex flex-row gap-2 items-center m-auto">
            <Textarea 
                placeholder="Ask Chatbot"
                className=" resize-none rounded-2xl bg-foreground/10 min-h-20 max-h-40 text-base focus:outline-none focus:ring-0 focus-visible:outline-none focus-visible:ring-0"
                value={query}
                onChange={event => setQuery(event.target.value)}
            />
            <div className="p-2 cursor-pointer rounded-full hover:bg-foreground/15" onClick={() => mutation.mutate({
                sessionId: activeSessionId,
                query: query
            })}>
                <Send size={20} />
            </div>
        </div>
    )
}