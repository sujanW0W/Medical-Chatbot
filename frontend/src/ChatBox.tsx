import type { Message } from "./types"
import { cn } from "./lib/utils"
import Markdown from "react-markdown"

export default function ChatBox({message}: {message: Message}) {
    return (
        <div className="py-8">
            <div className={cn("flex", message.role === "user" ? "justify-end w-3/4 max-w-fit ml-auto p-4 rounded-2xl bg-foreground/15" : "justify-start")}>
                <div className="markdown">
                {
                    <Markdown>
                        {message.content}
                    </Markdown>
                }
                </div>
            </div>
        </div>
    )
}