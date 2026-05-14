import { Sidebar, SidebarContent, SidebarFooter, SidebarHeader, SidebarGroup, SidebarTrigger, SidebarProvider } from "@/components/ui/sidebar";
import { Item, ItemContent, ItemActions, ItemDescription, ItemTitle } from "@/components/ui/item";
import { Ellipsis, SquarePen, LoaderCircle } from "lucide-react";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "./lib/utils";
import { useQuery } from "@tanstack/react-query";
import { fetchSessions } from "./queries";
import { useSession } from "./contexts/SessionContext";

function SidebarUI() {
    // const [sessions, setSessions] = useState<Array<Session>>([])
    const {activeSessionId, setActiveSessionId} = useSession()

    const {data: sessions, isLoading} = useQuery({queryKey: ["sessions"], queryFn: fetchSessions})
    
    return (
        <Sidebar className="[&>div:first-child]:overflow-y-auto [&>*]:overflow-x-hidden !border-r-0 bg-sidebar-background" collapsible="icon">
            <SidebarHeader className="sticky top-0 z-10 bg-sidebar-background shrink-0">
                <div className="flex flex-row items-center justify-between group-data-[collapsible=icon]:justify-center">
                    <p className="text-center text-2xl group-data-[collapsible=icon]:sidebar-collapsed-content sidebar-content">Medical Chatbot</p>
                    <SidebarTrigger 
                        className="[&>svg:first-child]:size-5"
                    />
                </div>
                <hr />
                <Item className="p-2 hoverable" onClick={() => setActiveSessionId(undefined)}>
                    <ItemContent className="flex flex-row gap-2 group-data-[collapsible=icon]:gap-0 items-center group-data-[collapsible=icon]:justify-center sidebar-content">
                        <SquarePen size={20} className="p-0" />
                        <p className="leading-normal group-data-[collapsible=icon]:sidebar-collapsed-content sidebar-content">New Chat</p>
                    </ItemContent>
                </Item>
                <p className="px-2 text-foreground/50 group-data-[collapsible=icon]:sidebar-collapsed-content sidebar-content">Chats</p>
            </SidebarHeader>
            <SidebarContent className="p-2 pb-0 overflow-visible relative bg-sidebar-background">
                {
                    <SidebarGroup className="p-0 flex-grow sidebar-content">
                    {
                        isLoading
                        ? <div className="flex justify-center">
                            <LoaderCircle size={20} className="animate-spin" />
                        </div>
                        : sessions && sessions.map(
                            session => (
                                <Item key={session.id} onClick={() => setActiveSessionId(session.id)} className={cn("p-2 hoverable group-data-[collapsible=icon]:sidebar-collapsed-content sidebar-content", session.id === activeSessionId && "focused")}>
                                    <ItemContent className="overflow-hidden">
                                        <p className="text-nowrap overflow-hidden text-ellipsis">{session.name}</p>
                                    </ItemContent>
                                    <ItemActions>
                                        <Ellipsis />
                                    </ItemActions>
                                </Item>
                            )
                        )
                    }
                </SidebarGroup>}
                <SidebarFooter className="sticky bottom-0 z-10 bg-sidebar-background p-0 pb-2 shrink-0">
                    <Item className="p-0">
                        <ItemContent className="flex flex-row gap-2 items-center overflow-hidden">
                            <Avatar className="m-1">
                                <AvatarImage 
                                    src="/assets/react.svb"
                                />
                                <AvatarFallback>
                                    SM
                                </AvatarFallback>
                            </Avatar>
                            <div className="flex-grow overflow-hidden group-data-[collapsible=icon]:hidden">
                            <ItemTitle className="inline-block w-full text-nowrap overflow-hidden text-ellipsis">React guy</ItemTitle>
                            <ItemDescription className="text-nowrap overflow-hidden text-ellipsis">
                                Free Tier
                            </ItemDescription>
                            </div>
                        </ItemContent>
                    </Item>
                </SidebarFooter>
            </SidebarContent>
        </Sidebar>
    )
}

export default function AppSidebar({children}: {children: React.ReactNode}) {
    
    return (
        <SidebarProvider>
            <SidebarUI />
            <>
                {children}
            </>
        </SidebarProvider>)
}