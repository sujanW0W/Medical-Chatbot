import ChatHeader from "./ChatHeader"
import ChatInput from "./ChatInput"
import ChatSection from "./ChatSection"

export default function Chat() {
    return (
        <div className="text-foreground flex-grow px-4 flex flex-col overflow-y-auto">
            <header className="sticky top-0 shrink-0 z-10 bg-background">
                <ChatHeader />
            </header>
            <div className="flex-grow flex flex-col px-8 lg:w-4/5 xl:w-3/5 m-auto">
                <div className="flex-grow flex flex-col pb-24 px-8">
                    <ChatSection />
                </div>
                <div className="sticky bottom-0 shrink-0 py-4 z-10 bg-background">
                    <ChatInput />
                </div>
            </div>
        </div>
    )
}