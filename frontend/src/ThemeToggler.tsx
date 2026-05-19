import { Moon, Sun } from "lucide-react";
import { useState, useEffect } from "react";

export default function ThemeToggler() {
    const [dark, setDark] = useState<boolean>(() => {
        return localStorage.getItem("theme") === "dark"
    })

    useEffect(() => {
        const root = document.documentElement
        
        if(dark) {
            localStorage.setItem("theme", "dark")
            root.classList.add("dark")
        } else{
            localStorage.setItem("theme", "light")
            root.classList.remove("dark")
        }
        
    }, [dark])

    return (
        <div onClick={() => setDark(prev => !prev)} className="cursor-pointer rounded-full hover:bg-foreground/15" >
            {
                dark ? (
                    <Moon size={20} />
                )
                :
                    <Sun size={20} />
            }
        </div>
    )
}