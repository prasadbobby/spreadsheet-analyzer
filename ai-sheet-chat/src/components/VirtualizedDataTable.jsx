'use client'

import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Table2, 
  Download, 
  Search, 
  X, 
  Filter,
  SortAsc,
  SortDesc,
  Eye,
  ArrowUpDown,
  Loader2
} from 'lucide-react'
import { formatCellValue } from '@/lib/utils'

const ROW_HEIGHT = 35
const DEFAULT_COLUMN_WIDTH = 150
const MIN_COLUMN_WIDTH = 100
const MAX_COLUMN_WIDTH = 400
const INITIAL_DISPLAY_ROWS = 100
const MODAL_CHUNK_SIZE = 1000

export default function OptimizedDataTable({ data, title = "Data Results" }) {
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' })
  const [columnWidths, setColumnWidths] = useState({})
  const [selectedRows, setSelectedRows] = useState(new Set())
  const [isProcessing, setIsProcessing] = useState(false)
  const [displayRowCount, setDisplayRowCount] = useState(INITIAL_DISPLAY_ROWS)
  const [modalPage, setModalPage] = useState(0)
  const containerRef = useRef(null)
  const modalContainerRef = useRef(null)
  
  if (!Array.isArray(data) || data.length === 0) return null

  const headers = Object.keys(data[0])

  // FIXED: Initialize column widths with useEffect to prevent re-renders
  useEffect(() => {
    if (Object.keys(columnWidths).length === 0) {
      const initialWidths = {}
      headers.forEach(header => {
        const headerWidth = Math.min(header.length * 8 + 40, MAX_COLUMN_WIDTH)
        const sampleValues = data.slice(0, 5).map(row => String(row[header] || ''))
        const maxValueWidth = Math.max(...sampleValues.map(val => Math.min(val.length * 7 + 20, MAX_COLUMN_WIDTH)))
        initialWidths[header] = Math.max(
          MIN_COLUMN_WIDTH, 
          Math.min(Math.max(headerWidth, maxValueWidth), MAX_COLUMN_WIDTH)
        )
      })
      setColumnWidths(initialWidths)
    }
  }, [headers, data, columnWidths])

  // Optimized data processing with memoization
  const processedData = useMemo(() => {
    setIsProcessing(true)
    
    let filtered = data

    if (searchTerm) {
      filtered = data.filter(row =>
        Object.values(row).some(value =>
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      )
    }

    if (sortConfig.key) {
      filtered = [...filtered].sort((a, b) => {
        const aVal = a[sortConfig.key]
        const bVal = b[sortConfig.key]
        
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal
        }
        
        const aStr = String(aVal || '').toLowerCase()
        const bStr = String(bVal || '').toLowerCase()
        
        if (aStr < bStr) return sortConfig.direction === 'asc' ? -1 : 1
        if (aStr > bStr) return sortConfig.direction === 'asc' ? 1 : -1
        return 0
      })
    }

    // Use setTimeout to prevent blocking
    setTimeout(() => setIsProcessing(false), 100)
    
    return filtered
  }, [data, searchTerm, sortConfig])

  // Display data with pagination for performance
  const displayData = useMemo(() => {
    return processedData.slice(0, displayRowCount)
  }, [processedData, displayRowCount])

  // Modal data with chunking
  const modalData = useMemo(() => {
    const start = modalPage * MODAL_CHUNK_SIZE
    const end = start + MODAL_CHUNK_SIZE
    return processedData.slice(start, end)
  }, [processedData, modalPage])

  const totalModalPages = Math.ceil(processedData.length / MODAL_CHUNK_SIZE)

  const handleSort = useCallback((column) => {
    setSortConfig(prev => ({
      key: column,
      direction: prev.key === column && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
    setDisplayRowCount(INITIAL_DISPLAY_ROWS) // Reset display count on sort
  }, [])

  const handleLoadMore = useCallback(() => {
    setDisplayRowCount(prev => Math.min(prev + 500, processedData.length))
  }, [processedData.length])

  const handleModalNext = useCallback(() => {
    if (modalPage < totalModalPages - 1) {
      setModalPage(prev => prev + 1)
    }
  }, [modalPage, totalModalPages])

  const handleModalPrev = useCallback(() => {
    if (modalPage > 0) {
      setModalPage(prev => prev - 1)
    }
  }, [modalPage])

  const exportToCSV = useCallback(() => {
    try {
      setIsProcessing(true)
      
      const csvHeaders = headers.join(',')
      const csvRows = processedData.map(row => 
        headers.map(header => {
          const value = row[header]
          const stringValue = String(value || '')
          if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
            return `"${stringValue.replace(/"/g, '""')}"`
          }
          return stringValue
        }).join(',')
      )
      
      const csvContent = [csvHeaders, ...csvRows].join('\n')
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
      const link = document.createElement('a')
      
      const url = URL.createObjectURL(blob)
      link.setAttribute('href', url)
      link.setAttribute('download', `data-export-${new Date().toISOString().slice(0, 10)}.csv`)
      link.style.visibility = 'hidden'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      setIsProcessing(false)
    } catch (error) {
      console.error('Error exporting CSV:', error)
      setIsProcessing(false)
    }
  }, [headers, processedData])

  // Optimized row component
  const TableRow = useCallback(({ row, rowIndex, isModal = false }) => (
    <tr
      key={rowIndex}
      className={`border-b border-gray-100 hover:bg-blue-50 cursor-pointer transition-colors ${
        selectedRows.has(rowIndex) ? 'bg-blue-100' : ''
      }`}
      onClick={() => {
        if (!isModal) {
          setSelectedRows(prev => {
            const newSet = new Set(prev)
            if (newSet.has(rowIndex)) {
              newSet.delete(rowIndex)
            } else {
              newSet.add(rowIndex)
            }
            return newSet
          })
        }
      }}
    >
      {headers.map((header) => (
        <td
          key={header}
          className="p-3 font-mono text-xs"
          style={{ 
            minWidth: columnWidths[header] || DEFAULT_COLUMN_WIDTH,
            maxWidth: columnWidths[header] || DEFAULT_COLUMN_WIDTH
          }}
        >
          <div className="truncate" title={String(row[header])}>
            {formatCellValue(row[header])}
          </div>
        </td>
      ))}
    </tr>
  ), [headers, columnWidths, selectedRows])

  // Header component
  const TableHeader = useCallback(({ isModal = false }) => (
    <thead className="bg-gray-50 sticky top-0 z-10">
      <tr className="border-b-2 border-gray-200">
        {headers.map((header) => (
          <th
            key={header}
            className={`text-left p-3 font-semibold text-gray-700 ${!isModal ? 'cursor-pointer hover:bg-gray-100' : ''}`}
            onClick={!isModal ? () => handleSort(header) : undefined}
            style={{ 
              minWidth: columnWidths[header] || DEFAULT_COLUMN_WIDTH,
              maxWidth: columnWidths[header] || DEFAULT_COLUMN_WIDTH
            }}
          >
            <div className="flex items-center justify-between">
              <span className="truncate" title={header}>{header}</span>
              {!isModal && (
                <div className="flex items-center gap-1">
                  {sortConfig.key === header && (
                    sortConfig.direction === 'asc' ? 
                    <SortAsc className="h-3 w-3" /> : 
                    <SortDesc className="h-3 w-3" />
                  )}
                  {sortConfig.key !== header && <ArrowUpDown className="h-3 w-3 opacity-30" />}
                </div>
              )}
            </div>
          </th>
        ))}
      </tr>
    </thead>
  ), [headers, columnWidths, sortConfig, handleSort])

  return (
    <>
      <Card className="border-0 shadow-lg bg-white">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <Table2 className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <CardTitle className="text-lg font-semibold text-gray-900">{title}</CardTitle>
                <p className="text-sm text-gray-600">
                  Showing {displayData.length.toLocaleString()} of {data.length.toLocaleString()} results
                  {isProcessing && <Loader2 className="h-3 w-3 animate-spin inline ml-2" />}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              {selectedRows.size > 0 && (
                <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                  {selectedRows.size} selected
                </Badge>
              )}
              <Button 
                size="sm" 
                variant="outline" 
                className="h-8"
                onClick={() => {
                  setViewModalOpen(true)
                  setModalPage(0)
                }}
                disabled={isProcessing}
              >
                <Eye className="h-3 w-3 mr-2" />
                View All
              </Button>
              <Button 
                size="sm" 
                variant="outline" 
                className="h-8"
                onClick={exportToCSV}
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                ) : (
                  <Download className="h-3 w-3 mr-2" />
                )}
                Export
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="p-0">
          {/* Search and Controls */}
          <div className="px-6 pb-4 flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search all columns..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 h-8"
                disabled={isProcessing}
              />
              {searchTerm && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0"
                  onClick={() => setSearchTerm('')}
                  disabled={isProcessing}
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>
            
            {selectedRows.size > 0 && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => setSelectedRows(new Set())}
                disabled={isProcessing}
              >
                Clear selection
              </Button>
            )}
          </div>

          {/* Optimized Table */}
          <div className="border rounded-lg mx-6 mb-6 overflow-hidden">
            <ScrollArea className="h-96 w-full" ref={containerRef}>
              <table className="w-full text-sm">
                <TableHeader />
                <tbody>
                  {displayData.map((row, rowIndex) => (
                    <TableRow key={rowIndex} row={row} rowIndex={rowIndex} />
                  ))}
                </tbody>
              </table>
            </ScrollArea>
          </div>

          {/* Load More Section */}
          {displayData.length < processedData.length && (
            <div className="mx-6 mb-6">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Filter className="h-5 w-5 text-blue-600" />
                    <div>
                      <p className="text-sm font-medium text-blue-800">
                        Showing {displayData.length.toLocaleString()} of {processedData.length.toLocaleString()} results
                      </p>
                      <p className="text-xs text-blue-600">
                        Load more rows or click "View All" for complete dataset
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="border-blue-300 text-blue-700 hover:bg-blue-100"
                      onClick={handleLoadMore}
                      disabled={isProcessing || displayData.length >= processedData.length}
                    >
                      Load More
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="border-blue-300 text-blue-700 hover:bg-blue-100"
                      onClick={() => {
                        setViewModalOpen(true)
                        setModalPage(0)
                      }}
                      disabled={isProcessing}
                    >
                      View All
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {processedData.length !== data.length && (
            <div className="mx-6 mb-6">
              <div className="bg-gradient-to-r from-orange-50 to-yellow-50 border border-orange-200 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <Filter className="h-5 w-5 text-orange-600" />
                  <div>
                    <p className="text-sm font-medium text-orange-800">
                      Filtered: {processedData.length.toLocaleString()} of {data.length.toLocaleString()} results
                    </p>
                    <p className="text-xs text-orange-600">
                      {searchTerm && `Search: "${searchTerm}"`}
                      {sortConfig.key && ` | Sorted by: ${sortConfig.key} (${sortConfig.direction})`}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Optimized Full Screen Modal */}
      <Dialog open={viewModalOpen} onOpenChange={setViewModalOpen}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] flex flex-col">
          <DialogHeader className="flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Table2 className="h-4 w-4 text-blue-600" />
                </div>
                <div>
                  <DialogTitle className="text-lg font-semibold">Complete Data View</DialogTitle>
                  <p className="text-sm text-gray-600">
                    {processedData.length.toLocaleString()} total results
                    {totalModalPages > 1 && (
                      <span className="ml-2">
                        (Page {modalPage + 1} of {totalModalPages})
                      </span>
                    )}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button size="sm" variant="outline" onClick={exportToCSV} disabled={isProcessing}>
                  {isProcessing ? (
                    <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                  ) : (
                    <Download className="h-3 w-3 mr-2" />
                  )}
                  Export CSV
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setViewModalOpen(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </DialogHeader>
          
          <div className="flex-1 min-h-0 mt-4 flex flex-col">
            {/* Pagination Controls */}
            {totalModalPages > 1 && (
              <div className="flex items-center justify-between mb-4 px-4 py-2 bg-gray-50 rounded-lg">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleModalPrev}
                  disabled={modalPage === 0}
                >
                  Previous
                </Button>
                <span className="text-sm text-gray-600">
                  Showing {modalPage * MODAL_CHUNK_SIZE + 1} - {Math.min((modalPage + 1) * MODAL_CHUNK_SIZE, processedData.length)} of {processedData.length.toLocaleString()}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleModalNext}
                  disabled={modalPage >= totalModalPages - 1}
                >
                  Next
                </Button>
              </div>
            )}

            <div className="flex-1 border rounded-lg overflow-hidden">
              <ScrollArea className="h-full w-full" ref={modalContainerRef}>
                <table className="w-full text-sm">
                  <TableHeader isModal={true} />
                  <tbody>
                    {modalData.map((row, rowIndex) => (
                      <TableRow 
                        key={modalPage * MODAL_CHUNK_SIZE + rowIndex} 
                        row={row} 
                        rowIndex={modalPage * MODAL_CHUNK_SIZE + rowIndex} 
                        isModal={true} 
                      />
                    ))}
                  </tbody>
                </table>
              </ScrollArea>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}