import AppSidebar from './SidebarUI'
import Chat from './Chat'
import "./App.css"
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SessionProvider } from '@/contexts/SessionProvider'
import { Toaster } from '@/components/ui/sonner'

const queryClient = new QueryClient()

function App() {
  return (
    <div className='w-full h-full flex flex-row'>
      <QueryClientProvider client={queryClient}>
        <SessionProvider>
          <AppSidebar>
            <Chat />
          </AppSidebar>
        </SessionProvider>
      </QueryClientProvider>
      <Toaster position="top-right" richColors closeButton />
    </div>
  )
}

export default App
