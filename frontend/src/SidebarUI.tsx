import { Sidebar, SidebarContent, SidebarFooter, SidebarHeader, SidebarGroup, SidebarTrigger, SidebarProvider } from "@/components/ui/sidebar";
import { Item, ItemContent, ItemActions, ItemDescription, ItemTitle } from "@/components/ui/item";
import { SquarePen, LoaderCircle, Pencil, Trash2 } from "lucide-react";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { cn } from "./lib/utils";
import { useRef, useState } from "react";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { fetchSessions, renameSession, deleteSession } from "./queries";
import { useSession } from "./contexts/SessionContext";
import type { Session } from "./types";

function SidebarUI() {
    const { activeSessionId, setActiveSessionId } = useSession()
    const queryClient = useQueryClient()

    const { data: sessions, isLoading } = useQuery({ queryKey: ["sessions"], queryFn: fetchSessions })

    const [renamingId, setRenamingId] = useState<string | undefined>(undefined)
    const [renameValue, setRenameValue] = useState("")
    const cancelingRef = useRef(false)
    const [deleteTarget, setDeleteTarget] = useState<Session | undefined>(undefined)

    const renameMutation = useMutation({
        mutationFn: renameSession,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["sessions"] })
        }
    })

    const deleteMutation = useMutation({
        mutationFn: deleteSession,
        onSuccess: (_data, sessionId) => {
            queryClient.invalidateQueries({ queryKey: ["sessions"] })
            if (activeSessionId === sessionId) {
                setActiveSessionId(undefined)
            }
            setDeleteTarget(undefined)
        }
    })

    const startRename = (session: Session) => {
        setRenamingId(session.id)
        setRenameValue(session.name)
    }

    const commitRename = (session: Session) => {
        const trimmed = renameValue.trim()
        if (trimmed && trimmed !== session.name) {
            renameMutation.mutate({ sessionId: session.id, name: trimmed })
        }
        setRenamingId(undefined)
    }

    return (
        <>
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
                                            <Item
                                                key={session.id}
                                                onClick={() => renamingId !== session.id && setActiveSessionId(session.id)}
                                                className={cn("p-2 hoverable group-data-[collapsible=icon]:sidebar-collapsed-content sidebar-content", session.id === activeSessionId && "focused")}
                                            >
                                                <ItemContent className="overflow-hidden">
                                                    {
                                                        renamingId === session.id
                                                            ? <Input
                                                                autoFocus
                                                                value={renameValue}
                                                                onClick={event => event.stopPropagation()}
                                                                onChange={event => setRenameValue(event.target.value)}
                                                                onKeyDown={event => {
                                                                    if (event.key === "Enter") {
                                                                        event.preventDefault()
                                                                        event.currentTarget.blur()
                                                                    } else if (event.key === "Escape") {
                                                                        event.preventDefault()
                                                                        cancelingRef.current = true
                                                                        setRenamingId(undefined)
                                                                    }
                                                                }}
                                                                onBlur={() => {
                                                                    if (cancelingRef.current) {
                                                                        cancelingRef.current = false
                                                                        return
                                                                    }
                                                                    commitRename(session)
                                                                }}
                                                                className="h-7 px-2 py-1 text-sm"
                                                            />
                                                            : <p className="text-nowrap overflow-hidden text-ellipsis">{session.name}</p>
                                                    }
                                                </ItemContent>
                                                {
                                                    renamingId !== session.id &&
                                                    <ItemActions className="opacity-0 group-hover/item:opacity-100 group-data-[collapsible=icon]:sidebar-collapsed-content sidebar-content">
                                                        <Pencil
                                                            size={16}
                                                            className="hover:text-foreground/70"
                                                            onClick={event => {
                                                                event.stopPropagation()
                                                                startRename(session)
                                                            }}
                                                        />
                                                        <Trash2
                                                            size={16}
                                                            className="hover:text-destructive"
                                                            onClick={event => {
                                                                event.stopPropagation()
                                                                setDeleteTarget(session)
                                                            }}
                                                        />
                                                    </ItemActions>
                                                }
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
            <Dialog open={!!deleteTarget} onOpenChange={open => !open && setDeleteTarget(undefined)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete chat?</DialogTitle>
                        <DialogDescription>
                            This will permanently delete "{deleteTarget?.name}". This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDeleteTarget(undefined)}>
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            disabled={deleteMutation.isPending}
                            onClick={() => deleteTarget && deleteMutation.mutate(deleteTarget.id)}
                        >
                            Delete
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    )
}

export default function AppSidebar({ children }: { children: React.ReactNode }) {

    return (
        <SidebarProvider>
            <SidebarUI />
            <>
                {children}
            </>
        </SidebarProvider>)
}
