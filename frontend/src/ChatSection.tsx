import ChatBox from "./ChatBox"
import { useEffect, useRef } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { fetchLatestJob, fetchMessages } from "./queries"
import { CircleAlert, LoaderCircle } from "lucide-react"
import { useSession } from "./contexts/SessionContext"

export default function ChatSection() {
    const { activeSessionId } = useSession()
    const queryClient = useQueryClient()

    const { data: job } = useQuery({
        queryKey: ["latest-job", activeSessionId],
        queryFn: () => fetchLatestJob(activeSessionId),
        enabled: !!activeSessionId,
        refetchInterval: (query) => {
            const jobStatus = query.state.data?.status
            return jobStatus === "queued" || jobStatus === "processing" ? 3_000 : false
        }
    })

    const { data, isLoading, isError, error } = useQuery({
        queryKey: ["messages", activeSessionId],
        queryFn: () => fetchMessages(activeSessionId),
        enabled: !!activeSessionId,
        refetchInterval: () => {
            return job?.status === "queued" || job?.status === "processing" ? 5_000 : false
        }
    })

    const prevJobStatusRef = useRef<string | undefined>(undefined)

    useEffect(() => {
        const prevStatus = prevJobStatusRef.current
        const currentStatus = job?.status

        if ((prevStatus === "queued" || prevStatus === "processing") && (currentStatus === "success" || currentStatus === "failed")) {
            queryClient.invalidateQueries({ queryKey: ["messages", activeSessionId] })
        }

        prevJobStatusRef.current = currentStatus
    }, [job?.status, activeSessionId, queryClient])

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
                        : <>
                            {data && data.map(msg => (
                                <ChatBox key={msg.id} message={msg} />
                            ))}
                            {(job?.status === "queued" || job?.status === "processing") && (
                                <div className="flex items-center gap-2 py-8 text-foreground/60">
                                    <LoaderCircle size={16} className="animate-spin" />
                                    <span className="text-sm">Thinking...</span>
                                </div>
                            )}
                        </>
            }
        </>
    )
}
