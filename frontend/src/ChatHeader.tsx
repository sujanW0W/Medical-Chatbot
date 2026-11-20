import { Ellipsis } from "lucide-react"
import { SidebarTrigger } from "./components/ui/sidebar"
import ThemeToggler from "./ThemeToggler"

export default function ChatHeader() {
    return (
        <div className="flex flex-row justify-between items-center p-2 border-b-[1px] border-foreground/15">
            <SidebarTrigger className="md:hidden [&>svg:first-child]:size-5" />
            <h1 className="text-2xl font-semibold">Medical Chatbot</h1>
            <div className="flex flex-row gap-4 [&>*]:p-1">
                <ThemeToggler />
                <div className="cursor-pointer">
                    <Ellipsis size={20} className="shrink-0" />
                </div>
            </div>
        </div>
    )
}