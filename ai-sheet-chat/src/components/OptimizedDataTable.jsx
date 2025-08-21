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
  Loader2,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Minimize2
} from 'lucide-react'
import { formatCellValue } from '@/lib/utils'

const MIN_COLUMN_WIDTH = 150
const MAX_COLUMN_WIDTH = 400
const INITIAL_DISPLAY_ROWS = 50
const MODAL_CHUNK_SIZE = 200 // Reduced for better performance
const DEBOUNCE_DELAY = 300

export default function OptimizedDataTable({ data, title = "Data Results" }) {
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [modalLoading, setModalLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('')
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' })
  const [columnWidths, setColumnWidths] = useState({})
  const [selectedRows, setSelectedRows] = useState(new Set())
  const [isProcessing, setIsProcessing] = useState(false)
  const [displayRowCount, setDisplayRowCount] = useState(INITIAL_DISPLAY_ROWS)
  const [modalPage, setModalPage] = useState(0)
  const [compactMode, setCompactMode] = useState(false)
  const [modalInitialized, setModalInitialized] = useState(false)
  
  // Refs for cleanup
  const debounceTimeoutRef = useRef(null)
  const processingTimeoutRef = useRef(null)
  
  if (!Array.isArray(data) || data.length === 0) return null

  const headers = Object.keys(data[0])

  // Debounced search to improve performance
  useEffect(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current)
    }
    
    debounceTimeoutRef.current = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm)
    }, DEBOUNCE_DELAY)

    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current)
      }
    }
  }, [searchTerm])

  // Optimize column width calculation
  const calculateOptimalWidth = useCallback((header, sampleData) => {
    const headerWidth = Math.min(header.length * 8 + 60, MAX_COLUMN_WIDTH)
    
    // Use smaller sample for performance
    const sampleSize = Math.min(10, sampleData.length)
    const sample = sampleData.slice(0, sampleSize)
    
    const contentWidths = sample.map(row => {
      const value = String(row[header] || '')
      const estimatedWidth = Math.min(value.length * 6 + 30, MAX_COLUMN_WIDTH)
      return estimatedWidth
    })
    
    const maxContentWidth = Math.max(...contentWidths, MIN_COLUMN_WIDTH)
    return Math.max(MIN_COLUMN_WIDTH, Math.min(Math.max(headerWidth, maxContentWidth), MAX_COLUMN_WIDTH))
  }, [])

  // Memoized column widths calculation
  const optimizedColumnWidths = useMemo(() => {
    if (data.length === 0) return {}
    
    const widths = {}
    const sampleData = data.slice(0, 5) // Use only 5 rows for width calculation
    
    headers.forEach(header => {
      widths[header] = calculateOptimalWidth(header, sampleData)
    })
    
    return widths
  }, [headers, data, calculateOptimalWidth])

  // Set column widths only once
  useEffect(() => {
    if (Object.keys(columnWidths).length === 0) {
      setColumnWidths(optimizedColumnWidths)
    }
  }, [optimizedColumnWidths, columnWidths])

  // Optimized data processing with better performance
  const processedData = useMemo(() => {
    let filtered = data

    if (debouncedSearchTerm) {
      filtered = data.filter(row =>
        Object.values(row).some(value =>
          String(value).toLowerCase().includes(debouncedSearchTerm.toLowerCase())
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

    return filtered
  }, [data, debouncedSearchTerm, sortConfig])

  // Memoized display data
  const displayData = useMemo(() => {
    return processedData.slice(0, displayRowCount)
  }, [processedData, displayRowCount])

  // Lazy modal data loading
  const modalData = useMemo(() => {
    if (!modalInitialized) return []
    const start = modalPage * MODAL_CHUNK_SIZE
    const end = start + MODAL_CHUNK_SIZE
    return processedData.slice(start, end)
  }, [processedData, modalPage, modalInitialized])

  const totalModalPages = Math.ceil(processedData.length / MODAL_CHUNK_SIZE)
  const totalTableWidth = headers.reduce((sum, header) => sum + (columnWidths[header] || MIN_COLUMN_WIDTH), 0)

  // Optimized handlers
  const handleSort = useCallback((column) => {
    setSortConfig(prev => ({
      key: column,
      direction: prev.key === column && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
    setDisplayRowCount(INITIAL_DISPLAY_ROWS)
  }, [])

  const handleLoadMore = useCallback(() => {
    setDisplayRowCount(prev => Math.min(prev + 100, processedData.length))
  }, [processedData.length])

  // Optimized modal open with loading state
  const handleOpenModal = useCallback(async () => {
    setModalLoading(true)
    setViewModalOpen(true)
    
    // Use requestAnimationFrame for smooth opening
    requestAnimationFrame(() => {
      setModalPage(0)
      setModalInitialized(true)
      
      // Slight delay to ensure smooth animation
      setTimeout(() => {
        setModalLoading(false)
      }, 100)
    })
  }, [])

  // Optimized modal close
  const handleCloseModal = useCallback(() => {
    setModalLoading(true)
    
    // Clean up modal state immediately
    setModalInitialized(false)
    
    // Close with slight delay for smooth animation
    requestAnimationFrame(() => {
      setViewModalOpen(false)
      setModalLoading(false)
    })
  }, [])

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

  const exportToCSV = useCallback(async () => {
    setIsProcessing(true)
    
    try {
      // Use setTimeout to prevent blocking UI
      await new Promise(resolve => setTimeout(resolve, 10))
      
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
      
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error exporting CSV:', error)
    } finally {
      setIsProcessing(false)
    }
  }, [headers, processedData])

  // Memoized row component for better performance
  const TableRow = useCallback(({ row, rowIndex, isModal = false }) => {
    const cellStyle = compactMode ? 'p-2 text-xs leading-tight' : 'p-3 text-sm leading-relaxed'
    
    return (
      <tr
        key={`row-${rowIndex}`}
        className={`border-b border-gray-200 hover:bg-blue-50 transition-colors ${
          selectedRows.has(rowIndex) ? 'bg-blue-100' : 'odd:bg-gray-50 even:bg-white'
        } ${!isModal ? 'cursor-pointer' : ''}`}
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
        {headers.map((header) => {
          const value = row[header]
          const formattedValue = formatCellValue(value)
          const cellWidth = columnWidths[header] || MIN_COLUMN_WIDTH
          
          return (
            <td
              key={`cell-${header}-${rowIndex}`}
              className="border-r border-gray-200 last:border-r-0 align-top"
              style={{ 
                width: cellWidth,
                minWidth: cellWidth,
                maxWidth: cellWidth
              }}
            >
              <div 
                className={cellStyle}
                style={{
                  wordWrap: 'break-word',
                  wordBreak: 'break-word',
                  whiteSpace: compactMode ? 'nowrap' : 'pre-wrap',
                  overflow: compactMode ? 'hidden' : 'visible',
                  textOverflow: compactMode ? 'ellipsis' : 'clip'
                }}
                title={String(value)}
              >
                <span className="font-mono">
                  {formattedValue}
                </span>
              </div>
            </td>
          )
        })}
      </tr>
    )
  }, [headers, columnWidths, selectedRows, compactMode])

  // Memoized header component
  const TableHeader = useCallback(({ isModal = false }) => (
    <thead className="bg-gradient-to-r from-gray-100 to-gray-200 sticky top-0 z-20 shadow-sm">
      <tr className="border-b-2 border-gray-300">
        {headers.map((header) => {
          const cellWidth = columnWidths[header] || MIN_COLUMN_WIDTH
          
          return (
            <th
              key={`header-${header}`}
              className={`text-left font-bold text-gray-800 border-r border-gray-300 last:border-r-0 ${!isModal ? 'cursor-pointer hover:bg-gray-200 transition-colors' : ''}`}
              onClick={!isModal ? () => handleSort(header) : undefined}
              style={{ 
                width: cellWidth,
                minWidth: cellWidth,
                maxWidth: cellWidth
              }}
            >
              <div className="p-3 flex items-center justify-between">
                <div className="flex-1 mr-2">
                  <div 
                    className="font-semibold text-sm"
                    style={{
                      wordWrap: 'break-word',
                      wordBreak: 'break-word',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}
                    title={header}
                  >
                    {header}
                  </div>
                </div>
                {!isModal && (
                  <div className="flex items-center gap-1 flex-shrink-0">
                    {sortConfig.key === header && (
                      sortConfig.direction === 'asc' ? 
                      <SortAsc className="h-4 w-4 text-blue-600" /> : 
                      <SortDesc className="h-4 w-4 text-blue-600" />
                    )}
                    {sortConfig.key !== header && <ArrowUpDown className="h-4 w-4 opacity-40" />}
                  </div>
                )}
              </div>
            </th>
          )
        })}
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
              
              {/* Optimized Compact Mode Toggle */}
              <Button
                size="sm"
                variant="outline"
                className="h-8"
                onClick={() => setCompactMode(!compactMode)}
              >
                {compactMode ? (
                  <Maximize2 className="h-3 w-3 mr-2" />
                ) : (
                  <Minimize2 className="h-3 w-3 mr-2" />
                )}
                {compactMode ? 'Expand' : 'Compact'}
              </Button>
              
              <Button 
                size="sm" 
                variant="outline" 
                className="h-8"
                onClick={handleOpenModal}
                disabled={modalLoading || isProcessing}
              >
                {modalLoading ? (
                  <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                ) : (
                  <Eye className="h-3 w-3 mr-2" />
                )}
                View All
              </Button>
              
              <Button 
                size="sm" 
                variant="outline" 
                className="h-8"
                onClick={exportToCSV}
                disabled={isProcessing || modalLoading}
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
          {/* Optimized Search */}
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

          {/* Info bar */}
          <div className="px-6 pb-4">
            <div className="flex items-center gap-4 text-xs text-gray-600">
              <div className="flex items-center gap-2">
                <Table2 className="h-3 w-3" />
                <span><strong>{headers.length}</strong> columns total</span>
              </div>
              <div className="flex items-center gap-2">
                <ArrowUpDown className="h-3 w-3" />
                <span>Scroll to explore • {compactMode ? 'Compact' : 'Full'} view</span>
              </div>
            </div>
          </div>

          {/* Optimized Main Table */}
          <div className="border rounded-lg mx-6 mb-6 overflow-hidden shadow-inner">
            <div 
              className="overflow-auto bg-white" 
              style={{ height: compactMode ? '500px' : '400px' }}
            >
              <div style={{ width: totalTableWidth, minWidth: '100%' }}>
                <table 
                  className="border-collapse"
                  style={{ 
                    width: totalTableWidth,
                    tableLayout: 'fixed'
                  }}
                >
                  <TableHeader />
                  <tbody>
                    {displayData.map((row, rowIndex) => (
                      <TableRow key={rowIndex} row={row} rowIndex={rowIndex} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
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
                        Optimized for performance • Load more as needed
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
                      onClick={handleOpenModal}
                      disabled={modalLoading || isProcessing}
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
                      {debouncedSearchTerm && `Search: "${debouncedSearchTerm}"`}
                      {sortConfig.key && ` | Sorted by: ${sortConfig.key} (${sortConfig.direction})`}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Optimized Modal */}
      <Dialog open={viewModalOpen} onOpenChange={handleCloseModal}>
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
                    {processedData.length.toLocaleString()} total • {headers.length} columns
                    {totalModalPages > 1 && (
                      <span className="ml-2">
                        (Page {modalPage + 1} of {totalModalPages})
                      </span>
                    )}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setCompactMode(!compactMode)}
                >
                  {compactMode ? <Maximize2 className="h-3 w-3 mr-2" /> : <Minimize2 className="h-3 w-3 mr-2" />}
                  {compactMode ? 'Expand' : 'Compact'}
                </Button>
                <Button size="sm" variant="outline" onClick={exportToCSV} disabled={isProcessing}>
                  {isProcessing ? <Loader2 className="h-3 w-3 mr-2 animate-spin" /> : <Download className="h-3 w-3 mr-2" />}
                  Export
                </Button>
                <Button size="sm" variant="ghost" onClick={handleCloseModal}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </DialogHeader>
          
          <div className="flex-1 min-h-0 mt-4 flex flex-col">
            {totalModalPages > 1 && (
              <div className="flex items-center justify-between mb-4 px-4 py-3 bg-gray-50 rounded-lg">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleModalPrev}
                  disabled={modalPage === 0}
                  className="flex items-center gap-2"
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous
                </Button>
                <div className="text-sm text-gray-600 font-medium">
                  Showing {modalPage * MODAL_CHUNK_SIZE + 1} - {Math.min((modalPage + 1) * MODAL_CHUNK_SIZE, processedData.length)} of {processedData.length.toLocaleString()}
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleModalNext}
                  disabled={modalPage >= totalModalPages - 1}
                  className="flex items-center gap-2"
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}

            {modalLoading ? (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-600" />
                  <p className="text-sm text-gray-600">Loading data...</p>
                </div>
              </div>
            ) : (
              <div className="flex-1 border rounded-lg overflow-hidden shadow-inner">
                <div className="overflow-auto h-full bg-white" style={{ maxHeight: '70vh' }}>
                  <div style={{ width: totalTableWidth, minWidth: '100%' }}>
                    <table 
                      className="border-collapse"
                      style={{ 
                        width: totalTableWidth,
                        tableLayout: 'fixed'
                      }}
                    >
                      <TableHeader isModal={true} />
                      <tbody>
                        {modalData.map((row, rowIndex) => (
                          <TableRow 
                            key={`modal-${modalPage}-${rowIndex}`}
                            row={row} 
                            rowIndex={modalPage * MODAL_CHUNK_SIZE + rowIndex} 
                            isModal={true} 
                          />
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}