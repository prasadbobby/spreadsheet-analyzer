import { useState } from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Table as TableIcon, Download, Eye, MoreHorizontal, X } from 'lucide-react'
import { formatCellValue } from '@/lib/utils'

export default function DataTable({ data }) {
  const [viewModalOpen, setViewModalOpen] = useState(false)

  if (!Array.isArray(data) || data.length === 0) return null

  const headers = Object.keys(data[0])
  const maxDisplayRows = 100
  const showRows = Math.min(data.length, maxDisplayRows)

  const exportToCSV = () => {
    try {
      // Create CSV content
      const csvHeaders = headers.join(',')
      const csvRows = data.map(row => 
        headers.map(header => {
          const value = row[header]
          // Escape quotes and wrap in quotes if contains comma
          const stringValue = String(value || '')
          if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
            return `"${stringValue.replace(/"/g, '""')}"`
          }
          return stringValue
        }).join(',')
      )
      
      const csvContent = [csvHeaders, ...csvRows].join('\n')
      
      // Create and trigger download
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
      const link = document.createElement('a')
      
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob)
        link.setAttribute('href', url)
        link.setAttribute('download', `data-export-${new Date().toISOString().slice(0, 10)}.csv`)
        link.style.visibility = 'hidden'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      }
    } catch (error) {
      console.error('Error exporting CSV:', error)
      alert('Error exporting data. Please try again.')
    }
  }

  return (
    <>
      <Card className="border-0 shadow-lg bg-white">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <TableIcon className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <CardTitle className="text-lg font-semibold text-gray-900">Data Results</CardTitle>
                <p className="text-sm text-gray-600">
                  Showing {showRows.toLocaleString()} of {data.length.toLocaleString()} results
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              {data.length > maxDisplayRows && (
                <Badge variant="secondary" className="bg-orange-100 text-orange-800">
                  Truncated for performance
                </Badge>
              )}
              <Button 
                size="sm" 
                variant="outline" 
                className="h-8"
                onClick={() => setViewModalOpen(true)}
              >
                <Eye className="h-3 w-3 mr-2" />
                View
              </Button>
              <Button 
                size="sm" 
                variant="outline" 
                className="h-8"
                onClick={exportToCSV}
              >
                <Download className="h-3 w-3 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="p-0">
          <div className="border rounded-lg mx-6 mb-6 overflow-hidden">
            <ScrollArea className="h-96 w-full">
              <Table className="data-table">
                <TableHeader>
                  <TableRow className="bg-gray-50 hover:bg-gray-50">
                    {headers.map((header, index) => (
                      <TableHead key={index} className="font-semibold text-gray-700 bg-gray-50 sticky top-0 z-10 border-b-2 border-gray-200">
                        <div className="flex items-center gap-2">
                          <span>{header}</span>
                          <MoreHorizontal className="h-3 w-3 text-gray-400" />
                        </div>
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.slice(0, showRows).map((row, rowIndex) => (
                    <TableRow key={rowIndex} className="hover:bg-blue-50/50 transition-colors">
                      {headers.map((header, cellIndex) => (
                        <TableCell key={cellIndex} className="font-mono text-sm border-b border-gray-100">
                          <div className="max-w-xs truncate" title={String(row[header])}>
                            {formatCellValue(row[header])}
                          </div>
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </div>

          {data.length > showRows && (
            <div className="mx-6 mb-6">
              <div className="bg-gradient-to-r from-orange-50 to-yellow-50 border border-orange-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Eye className="h-5 w-5 text-orange-600" />
                    <div>
                      <p className="text-sm font-medium text-orange-800">
                        Complete dataset has {data.length.toLocaleString()} results
                      </p>
                      <p className="text-xs text-orange-600">
                        Table shows first {showRows.toLocaleString()} rows for optimal performance
                      </p>
                    </div>
                  </div>
                  <Button 
                    size="sm" 
                    variant="outline" 
                    className="border-orange-300 text-orange-700 hover:bg-orange-100"
                    onClick={() => setViewModalOpen(true)}
                  >
                    View All
                  </Button>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* View Modal */}
      <Dialog open={viewModalOpen} onOpenChange={setViewModalOpen}>
        <DialogContent className="max-w-6xl max-h-[90vh] flex flex-col">
          <DialogHeader className="flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <TableIcon className="h-4 w-4 text-blue-600" />
                </div>
                <div>
                  <DialogTitle className="text-lg font-semibold">Complete Data View</DialogTitle>
                  <p className="text-sm text-gray-600">
                    All {data.length.toLocaleString()} results
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={exportToCSV}
                >
                  <Download className="h-3 w-3 mr-2" />
                  Export CSV
                </Button>
                <Button 
                  size="sm" 
                  variant="ghost" 
                  onClick={() => setViewModalOpen(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </DialogHeader>
          
          <div className="flex-1 min-h-0 mt-4">
            <div className="border rounded-lg overflow-hidden h-full">
              <ScrollArea className="h-full w-full">
                <Table className="data-table">
                  <TableHeader>
                    <TableRow className="bg-gray-50 hover:bg-gray-50">
                      {headers.map((header, index) => (
                        <TableHead key={index} className="font-semibold text-gray-700 bg-gray-50 sticky top-0 z-10 border-b-2 border-gray-200">
                          <div className="flex items-center gap-2">
                            <span>{header}</span>
                          </div>
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.map((row, rowIndex) => (
                      <TableRow key={rowIndex} className="hover:bg-blue-50/50 transition-colors">
                        {headers.map((header, cellIndex) => (
                          <TableCell key={cellIndex} className="font-mono text-sm border-b border-gray-100">
                            <div className="max-w-xs truncate" title={String(row[header])}>
                              {formatCellValue(row[header])}
                            </div>
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}