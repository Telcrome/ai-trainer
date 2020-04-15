import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { Log } from './log';
import { ConfigService } from '../config.service';

@Injectable({
  providedIn: 'root'
})
export class LogService {

  constructor(private http: HttpClient, private config: ConfigService) { }

  getLogs(): Observable<any> {
    const endpointString = `${this.config.serverCreds.url}/logs/`;
    return this.http.get(endpointString);
  }
}
