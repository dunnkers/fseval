import React, { useState, useRef, useEffect } from 'react';
import FileTree from './FileTree';
import CodeBlock from '@theme/CodeBlock';

const DEFAULT_ITEM = {
    data: {
        filePath: '',
        fileExtension: '',
        content: '',
    }
};

export default function FileTreeCodeViewer({
    template,
    treeId,
    viewState,
    defaultItem = DEFAULT_ITEM
}) {
    const environment = useRef();
    const [item, setItem] = useState(defaultItem);
    const itemData = (item || DEFAULT_ITEM).data;

    // simulate clicking the default value.
    useEffect(() => {
        try {
            const treeElement = environment.current;
            const treeEnvironmentContext = treeElement.treeEnvironmentContext;
            const viewState = treeEnvironmentContext.viewState;
            const viewStateTree = viewState[treeId];
            const selectedItems = viewStateTree.selectedItems;

            for (const selectedItem of selectedItems) {
                if (!selectedItem) continue;

                treeElement.invokePrimaryAction(selectedItem, treeId)
            }
        } catch (error) {
            return; // fail softly
        }
    });

    return (
        <div className="row">
            <div className="col col--4">
                <FileTree
                    template={template}
                    treeId={treeId}
                    viewState={viewState}
                    onPrimaryAction={setItem}
                    environment={environment}
                />
            </div>
            <div className="col col--8">
                <CodeBlock
                    className={`language-${itemData.fileExtension}`}
                    title={itemData.filePath}
                >
                    {itemData.content}
                </CodeBlock>
            </div>
        </div>
    )
}