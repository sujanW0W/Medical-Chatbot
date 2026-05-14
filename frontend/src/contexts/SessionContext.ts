import { createContext, useContext } from "react";
import type { Dispatch, SetStateAction } from "react";

export interface SessionContextType {
    activeSessionId: string | undefined;
    setActiveSessionId: Dispatch<SetStateAction<string | undefined>>;
}

export const SessionContext = createContext<SessionContextType>({
    activeSessionId: undefined,
    setActiveSessionId: () => {},
});

export const useSession = () => {
    const context = useContext(SessionContext);

    if (!context) {
        throw new Error("Context Error");
    }

    return context;
};
