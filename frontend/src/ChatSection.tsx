import ChatBox from "./ChatBox"
import { useQuery } from "@tanstack/react-query"
import { fetchMessages } from "./queries"
import { LoaderCircle } from "lucide-react"
import { useSession } from "./contexts/SessionContext"

export default function ChatSection() {
    const {activeSessionId} = useSession()

    const {data, isLoading} = useQuery({queryKey:["messages", activeSessionId], queryFn: () => fetchMessages(activeSessionId)})

    return (
        <>
            {
                isLoading
                ? <div className="w-full h-full flex justify-center items-center">
                    <LoaderCircle size={20} className="animate-spin" />
                </div>
                : data && data.map(msg => (
                    <ChatBox key={msg.id} message={msg} />

                ))
            }
        </>
    )
}