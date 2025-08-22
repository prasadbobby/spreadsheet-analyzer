'use client'

import { createContext, useContext } from 'react'
import { useSidebar } from '@/hooks/useSidebar'

const SidebarContext = createContext({})

export const useSidebarContext = () => {
  const context = useContext(SidebarContext)
  if (!context) {
    throw new Error('useSidebarContext must be used within SidebarProvider')
  }
  return context
}

export const SidebarProvider = ({ children }) => {
  const sidebar = useSidebar()

  return (
    <SidebarContext.Provider value={sidebar}>
      {children}
    </SidebarContext.Provider>
  )
}