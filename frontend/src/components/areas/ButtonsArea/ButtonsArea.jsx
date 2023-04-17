import React from 'react'
import SaveButton from '../../buttons/SaveButton/SaveButton'
import ExportButton from '../../buttons/ExportButton/ExportButton'

export default function ButtonsArea() {
    return (
    <div className='buttons-area'>
        <SaveButton />
        <ExportButton />
        <span>More content soon (no)</span>
    </div>
    )
}
