import ChatBox from "./ChatBox"
import { useQuery } from "@tanstack/react-query"
import { fetchMessages } from "./queries"
import { CircleAlert, LoaderCircle } from "lucide-react"
import { useSession } from "./contexts/SessionContext"

export default function ChatSection() {
    const { activeSessionId } = useSession()

    const { data, isLoading, isError, error } = useQuery({
        queryKey: ["messages", activeSessionId],
        queryFn: () => fetchMessages(activeSessionId),
        enabled: !!activeSessionId,
        refetchInterval: (query) => {
            if (!activeSessionId) return false
            if (query.state.error) return false
            const lastRole = query.state.data?.at(-1)?.role
            return lastRole == "user" ? 5_000 : false
        }
    })

    return (
        <>
            {
                isLoading
                    ? <div className="w-full h-full flex justify-center items-center">
                        <LoaderCircle size={20} className="animate-spin" />
                    </div>
                    : isError && !data
                        ? <div className="w-full h-full flex flex-col gap-2 items-center justify-center text-center text-foreground/60">
                            <CircleAlert size={20} />
                            <p className="text-sm">{error?.message || "Couldn't load messages. Please try again."}</p>
                        </div>
                        : data && data.map(msg => (
                            <ChatBox key={msg.id} message={msg} />

                        ))
            }
        </>
    )
}