import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { photoData, type ChatPhoto } from '../lib/chat-photos';
import { IconBtn } from './ui';
export function ChatPhotos({ photos, onRemove }: {
    photos: ChatPhoto[];
    onRemove?: (id: string) => void;
}) {
    const [images, setImages] = useState<{
        id: string;
        name: string;
        url: string;
    }[]>([]);
    useEffect(() => {
        let current = true;
        void Promise.all(photos.map(async (photo) => ({ ...await photoData(photo), id: photo.id }))).then(value => {
            if (current)
                setImages(value);
        }).catch(() => {
            if (current)
                setImages([]);
        });
        return () => { current = false; };
    }, [photos]);
    return <div className="chat-photos">{tr(images.map(i => <div key={i.id}><img src={i.url} alt={i.name}/><span>{i.name}</span>{tr(onRemove ? <IconBtn name="x" title={tr("Remove {0}", [i.name])} onClick={() => onRemove(i.id)}/> : null)}</div>))}</div>;
}
