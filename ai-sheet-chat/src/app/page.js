'use client'

import { useState, useEffect } from 'react'
import Sidebar from '@/components/Sidebar'
import MainContent from '@/components/MainContent'
import { AISheetChatProvider } from '@/contexts/AISheetChatContext'

export default function Home() {
  return (
    <AISheetChatProvider>
      <div className="flex h-screen overflow-hidden bg-background">
        <Sidebar />
        <MainContent />
      </div>
    </AISheetChatProvider>
  )
}