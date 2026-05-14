import { useState, useEffect } from "react"
import { SessionContext } from "@/contexts/SessionContext"

export const SessionProvider = ({children}: {children: React.ReactElement}) => {
    const [activeSessionId, setActiveSessionId] = useState(() => {
        const savedSessionId = localStorage.getItem("activeSessionId")
        return savedSessionId || undefined
    })

    useEffect(() => {
        localStorage.setItem("activeSessionId", activeSessionId || "")
    }, [activeSessionId])

    return (
        <SessionContext.Provider value={{activeSessionId, setActiveSessionId}}>
            {children}
        </SessionContext.Provider>
    )
}