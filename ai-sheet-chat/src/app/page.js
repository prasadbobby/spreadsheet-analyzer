'use client'

import Sidebar from '@/components/Sidebar'
import MainContent from '@/components/MainContent'
import { AISheetChatProvider } from '@/contexts/AISheetChatContext'
import { SidebarProvider } from '@/contexts/SidebarContext'

export default function Home() {
  return (
    <AISheetChatProvider>
      <SidebarProvider>
        <div className="flex h-screen overflow-hidden bg-background">
          <Sidebar />
          <MainContent />
        </div>
      </SidebarProvider>
    </AISheetChatProvider>
  )
}